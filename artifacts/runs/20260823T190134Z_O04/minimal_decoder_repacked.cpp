#include "h40/attention.hpp"
#include "h40/expert_cache.hpp"
#include "h40/expert_loader.hpp"
#include "h40/flash_tensor_provider.hpp"
#include "h40/gptoss_expert.hpp"
#include "h40/h40m_tensor_catalog.hpp"
#include "h40/model_index.hpp"
#include "h40/moe_scheduler.hpp"
#include "h40/parallel_bf16.hpp"
#include "h40/trace.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <span>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {

constexpr std::size_t kLayers = 24;
constexpr std::size_t kHidden = 2880;
constexpr std::size_t kIntermediate = 2880;
constexpr std::size_t kExperts = 32;
constexpr std::size_t kTopK = 4;
constexpr std::size_t kQHeads = 64;
constexpr std::size_t kKvHeads = 8;
constexpr std::size_t kHeadDim = 64;
constexpr std::size_t kQDim = kQHeads * kHeadDim;
constexpr std::size_t kKvDim = kKvHeads * kHeadDim;
constexpr std::size_t kVocab = 201088;
constexpr std::size_t kExpertPayloadBytes = 13236480;
constexpr std::size_t kExpertStrideBytes = 13631488;
constexpr std::size_t kLmHeadChunkRows = 8192;
constexpr std::size_t kMaxDenseThreads = 8;
constexpr std::size_t kDefaultDenseThreads = 6;

enum class ExpertReuseMode {
    off,
    exact,
    approximate,
};

const char* reuse_mode_name(ExpertReuseMode mode) {
    switch (mode) {
        case ExpertReuseMode::off: return "off";
        case ExpertReuseMode::exact: return "exact";
        case ExpertReuseMode::approximate: return "approximate";
    }
    return "unknown";
}

struct Metrics {
    std::uint64_t dense_bytes{};
    std::uint64_t expert_flash_bytes{};
    std::uint64_t expert_cache_hits{};
    std::uint64_t expert_cache_misses{};
    std::uint64_t layers_run{};
    std::uint32_t token_id{};
    float token_logit{-std::numeric_limits<float>::infinity()};
    std::uint64_t peak_rss_kib{};
    std::uint64_t prefetched_experts{};
    std::uint64_t prefetch_read_ns{};
    std::uint64_t prefetch_wait_ns{};
    std::uint64_t embedding_ns{};
    std::uint64_t dense_matvec_ns{};
    std::uint64_t attention_ns{};
    std::uint64_t moe_ns{};
    std::uint64_t lm_head_ns{};
    std::uint64_t expert_reuse_hits{};
    std::uint64_t expert_reuse_misses{};
    std::size_t dense_threads{1};
    std::size_t expert_reuse_window{};
    std::string cache_policy{"lru"};
    std::string expert_reuse_mode{"off"};
    bool io_overlap_enabled{};
};

std::uint64_t elapsed_ms(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count());
}

std::vector<std::uint32_t> parse_tokens(std::string_view text) {
    std::vector<std::uint32_t> tokens;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto comma = text.find(',', start);
        const auto part = text.substr(start, comma == std::string_view::npos ? text.size() - start : comma - start);
        if (!part.empty()) tokens.push_back(static_cast<std::uint32_t>(std::stoul(std::string(part))));
        if (comma == std::string_view::npos) break;
        start = comma + 1;
    }
    if (tokens.empty()) throw std::invalid_argument("at least one token id is required");
    return tokens;
}

h40::H40mTensorRecord must_find(const h40::H40mTensorCatalog& catalog, const std::string& name) {
    auto record = catalog.find(name);
    if (!record.has_value()) throw std::runtime_error("missing tensor: " + name);
    return *record;
}

void add_bias(std::span<float> values, std::span<const float> bias) {
    if (values.size() != bias.size()) throw std::invalid_argument("bias size mismatch");
    for (std::size_t i = 0; i < values.size(); ++i) values[i] += bias[i];
}

void add_inplace(std::span<float> lhs, std::span<const float> rhs) {
    if (lhs.size() != rhs.size()) throw std::invalid_argument("residual size mismatch");
    for (std::size_t i = 0; i < lhs.size(); ++i) lhs[i] += rhs[i];
}

void bf16_matvec_counted(
    h40::ParallelBf16Matvec& executor,
    const h40::H40mTensorRecord& record,
    std::span<const float> input,
    std::span<float> output,
    std::size_t workers,
    Metrics& metrics) {
    const auto start = std::chrono::steady_clock::now();
    executor.matvec(record, input, output, workers);
    metrics.dense_matvec_ns += static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
    metrics.dense_bytes += record.length;
}

void bf16_vector_counted(
    const h40::FileTensorReader& reader,
    const h40::H40mTensorRecord& record,
    std::span<float> output,
    Metrics& metrics) {
    reader.read_bf16_vector(record, output);
    metrics.dense_bytes += record.length;
}

void bf16_row_counted(
    const h40::FileTensorReader& reader,
    const h40::H40mTensorRecord& record,
    std::size_t row,
    std::span<float> output,
    Metrics& metrics) {
    reader.read_bf16_row(record, row, output);
    metrics.dense_bytes += record.shape[1] * sizeof(std::uint16_t);
}

h40::ModelIndex build_expert_index() {
    static constexpr std::uint64_t kRepackedOffsets[kLayers * kExperts] = {
        231735296ULL,
        299892736ULL,
        68157440ULL,
        218103808ULL,
        81788928ULL,
        27262976ULL,
        13631488ULL,
        136314880ULL,
        204472320ULL,
        109051904ULL,
        327155712ULL,
        54525952ULL,
        163577856ULL,
        177209344ULL,
        408944640ULL,
        40894464ULL,
        122683392ULL,
        272629760ULL,
        95420416ULL,
        245366784ULL,
        368050176ULL,
        381681664ULL,
        258998272ULL,
        313524224ULL,
        0ULL,
        190840832ULL,
        354418688ULL,
        149946368ULL,
        286261248ULL,
        395313152ULL,
        340787200ULL,
        422576128ULL,
        558891008ULL,
        531628032ULL,
        599785472ULL,
        736100352ULL,
        477102080ULL,
        517996544ULL,
        804257792ULL,
        708837376ULL,
        449839104ULL,
        831520768ULL,
        572522496ULL,
        490733568ULL,
        640679936ULL,
        776994816ULL,
        858783744ULL,
        845152256ULL,
        586153984ULL,
        463470592ULL,
        749731840ULL,
        504365056ULL,
        613416960ULL,
        667942912ULL,
        695205888ULL,
        817889280ULL,
        722468864ULL,
        627048448ULL,
        790626304ULL,
        654311424ULL,
        545259520ULL,
        681574400ULL,
        763363328ULL,
        436207616ULL,
        1145044992ULL,
        926941184ULL,
        899678208ULL,
        1104150528ULL,
        1131413504ULL,
        1049624576ULL,
        1076887552ULL,
        872415232ULL,
        1199570944ULL,
        1281359872ULL,
        1063256064ULL,
        913309696ULL,
        1022361600ULL,
        1008730112ULL,
        940572672ULL,
        886046720ULL,
        995098624ULL,
        954204160ULL,
        1294991360ULL,
        981467136ULL,
        1172307968ULL,
        1090519040ULL,
        1035993088ULL,
        1240465408ULL,
        1158676480ULL,
        1254096896ULL,
        1267728384ULL,
        1185939456ULL,
        1226833920ULL,
        1213202432ULL,
        967835648ULL,
        1117782016ULL,
        1404043264ULL,
        1444937728ULL,
        1567621120ULL,
        1676673024ULL,
        1540358144ULL,
        1717567488ULL,
        1703936000ULL,
        1431306240ULL,
        1663041536ULL,
        1635778560ULL,
        1322254336ULL,
        1390411776ULL,
        1485832192ULL,
        1513095168ULL,
        1690304512ULL,
        1335885824ULL,
        1376780288ULL,
        1526726656ULL,
        1731198976ULL,
        1417674752ULL,
        1349517312ULL,
        1608515584ULL,
        1499463680ULL,
        1594884096ULL,
        1472200704ULL,
        1458569216ULL,
        1553989632ULL,
        1363148800ULL,
        1308622848ULL,
        1622147072ULL,
        1649410048ULL,
        1581252608ULL,
        2112880640ULL,
        2140143616ULL,
        1840250880ULL,
        2017460224ULL,
        2031091712ULL,
        2003828736ULL,
        1949302784ULL,
        1826619392ULL,
        1935671296ULL,
        1772093440ULL,
        1908408320ULL,
        1812987904ULL,
        1785724928ULL,
        1881145344ULL,
        2153775104ULL,
        1976565760ULL,
        1867513856ULL,
        2071986176ULL,
        1962934272ULL,
        2167406592ULL,
        1894776832ULL,
        1758461952ULL,
        2044723200ULL,
        2085617664ULL,
        1990197248ULL,
        2058354688ULL,
        2126512128ULL,
        2099249152ULL,
        1799356416ULL,
        1744830464ULL,
        1922039808ULL,
        1853882368ULL,
        2290089984ULL,
        2603614208ULL,
        2440036352ULL,
        2330984448ULL,
        2562719744ULL,
        2576351232ULL,
        2371878912ULL,
        2181038080ULL,
        2262827008ULL,
        2453667840ULL,
        2589982720ULL,
        2494562304ULL,
        2276458496ULL,
        2549088256ULL,
        2303721472ULL,
        2385510400ULL,
        2344615936ULL,
        2412773376ULL,
        2426404864ULL,
        2317352960ULL,
        2235564032ULL,
        2535456768ULL,
        2521825280ULL,
        2208301056ULL,
        2358247424ULL,
        2399141888ULL,
        2194669568ULL,
        2249195520ULL,
        2221932544ULL,
        2508193792ULL,
        2480930816ULL,
        2467299328ULL,
        2658140160ULL,
        2958032896ULL,
        2848980992ULL,
        2712666112ULL,
        2780823552ULL,
        2835349504ULL,
        2767192064ULL,
        2617245696ULL,
        3012558848ULL,
        2876243968ULL,
        2821718016ULL,
        2699034624ULL,
        2739929088ULL,
        2998927360ULL,
        2630877184ULL,
        2889875456ULL,
        2794455040ULL,
        2726297600ULL,
        2917138432ULL,
        2944401408ULL,
        2753560576ULL,
        2862612480ULL,
        2644508672ULL,
        2903506944ULL,
        2671771648ULL,
        2985295872ULL,
        3039821824ULL,
        2930769920ULL,
        2685403136ULL,
        2971664384ULL,
        3026190336ULL,
        2808086528ULL,
        3121610752ULL,
        3326083072ULL,
        3271557120ULL,
        3285188608ULL,
        3176136704ULL,
        3244294144ULL,
        3407872000ULL,
        3448766464ULL,
        3380609024ULL,
        3189768192ULL,
        3203399680ULL,
        3107979264ULL,
        3462397952ULL,
        3366977536ULL,
        3421503488ULL,
        3067084800ULL,
        3394240512ULL,
        3312451584ULL,
        3135242240ULL,
        3353346048ULL,
        3162505216ULL,
        3148873728ULL,
        3094347776ULL,
        3230662656ULL,
        3339714560ULL,
        3435134976ULL,
        3080716288ULL,
        3053453312ULL,
        3298820096ULL,
        3257925632ULL,
        3217031168ULL,
        3476029440ULL,
        3680501760ULL,
        3598712832ULL,
        3585081344ULL,
        3639607296ULL,
        3721396224ULL,
        3544186880ULL,
        3625975808ULL,
        3666870272ULL,
        3735027712ULL,
        3857711104ULL,
        3789553664ULL,
        3489660928ULL,
        3571449856ULL,
        3694133248ULL,
        3844079616ULL,
        3898605568ULL,
        3503292416ULL,
        3707764736ULL,
        3775922176ULL,
        3557818368ULL,
        3762290688ULL,
        3530555392ULL,
        3612344320ULL,
        3748659200ULL,
        3653238784ULL,
        3830448128ULL,
        3816816640ULL,
        3912237056ULL,
        3516923904ULL,
        3803185152ULL,
        3871342592ULL,
        3884974080ULL,
        4048551936ULL,
        3980394496ULL,
        4171235328ULL,
        4280287232ULL,
        4075814912ULL,
        4225761280ULL,
        3939500032ULL,
        4307550208ULL,
        4143972352ULL,
        3966763008ULL,
        4348444672ULL,
        4034920448ULL,
        3953131520ULL,
        4253024256ULL,
        4334813184ULL,
        4130340864ULL,
        4157603840ULL,
        3925868544ULL,
        4103077888ULL,
        4212129792ULL,
        4007657472ULL,
        4239392768ULL,
        4266655744ULL,
        4184866816ULL,
        4116709376ULL,
        4089446400ULL,
        3994025984ULL,
        4293918720ULL,
        4021288960ULL,
        4321181696ULL,
        4198498304ULL,
        4062183424ULL,
        4771020800ULL,
        4375707648ULL,
        4389339136ULL,
        4512022528ULL,
        4743757824ULL,
        4580179968ULL,
        4702863360ULL,
        4675600384ULL,
        4566548480ULL,
        4416602112ULL,
        4648337408ULL,
        4402970624ULL,
        4498391040ULL,
        4362076160ULL,
        4607442944ULL,
        4471128064ULL,
        4443865088ULL,
        4784652288ULL,
        4621074432ULL,
        4634705920ULL,
        4730126336ULL,
        4484759552ULL,
        4716494848ULL,
        4593811456ULL,
        4525654016ULL,
        4661968896ULL,
        4689231872ULL,
        4457496576ULL,
        4552916992ULL,
        4539285504ULL,
        4430233600ULL,
        4757389312ULL,
        4798283776ULL,
        5179965440ULL,
        5070913536ULL,
        4907335680ULL,
        5193596928ULL,
        5002756096ULL,
        4866441216ULL,
        5043650560ULL,
        5057282048ULL,
        5139070976ULL,
        4839178240ULL,
        5220859904ULL,
        5030019072ULL,
        4989124608ULL,
        4880072704ULL,
        5125439488ULL,
        5166333952ULL,
        4811915264ULL,
        4961861632ULL,
        4920967168ULL,
        4934598656ULL,
        5152702464ULL,
        5016387584ULL,
        4825546752ULL,
        5111808000ULL,
        5084545024ULL,
        4948230144ULL,
        5207228416ULL,
        4893704192ULL,
        4852809728ULL,
        5098176512ULL,
        4975493120ULL,
        5588910080ULL,
        5289017344ULL,
        5425332224ULL,
        5384437760ULL,
        5234491392ULL,
        5466226688ULL,
        5357174784ULL,
        5316280320ULL,
        5329911808ULL,
        5520752640ULL,
        5575278592ULL,
        5643436032ULL,
        5534384128ULL,
        5438963712ULL,
        5507121152ULL,
        5398069248ULL,
        5479858176ULL,
        5616173056ULL,
        5452595200ULL,
        5602541568ULL,
        5657067520ULL,
        5343543296ULL,
        5629804544ULL,
        5302648832ULL,
        5548015616ULL,
        5411700736ULL,
        5261754368ULL,
        5493489664ULL,
        5248122880ULL,
        5370806272ULL,
        5275385856ULL,
        5561647104ULL,
        5697961984ULL,
        5861539840ULL,
        6025117696ULL,
        5997854720ULL,
        5929697280ULL,
        5752487936ULL,
        5820645376ULL,
        5875171328ULL,
        5711593472ULL,
        5916065792ULL,
        5793382400ULL,
        6066012160ULL,
        5847908352ULL,
        6079643648ULL,
        5956960256ULL,
        5970591744ULL,
        5888802816ULL,
        5725224960ULL,
        6038749184ULL,
        5902434304ULL,
        5779750912ULL,
        5834276864ULL,
        5984223232ULL,
        5670699008ULL,
        5766119424ULL,
        6011486208ULL,
        5684330496ULL,
        5943328768ULL,
        6052380672ULL,
        5738856448ULL,
        5807013888ULL,
        6093275136ULL,
        6461325312ULL,
        6175064064ULL,
        6488588288ULL,
        6311378944ULL,
        6161432576ULL,
        6352273408ULL,
        6215958528ULL,
        6502219776ULL,
        6120538112ULL,
        6256852992ULL,
        6474956800ULL,
        6379536384ULL,
        6420430848ULL,
        6529482752ULL,
        6434062336ULL,
        6147801088ULL,
        6229590016ULL,
        6284115968ULL,
        6243221504ULL,
        6447693824ULL,
        6393167872ULL,
        6515851264ULL,
        6338641920ULL,
        6188695552ULL,
        6297747456ULL,
        6365904896ULL,
        6106906624ULL,
        6270484480ULL,
        6134169600ULL,
        6202327040ULL,
        6406799360ULL,
        6325010432ULL,
        6829375488ULL,
        6788481024ULL,
        6570377216ULL,
        6706692096ULL,
        6883901440ULL,
        6665797632ULL,
        6897532928ULL,
        6652166144ULL,
        6815744000ULL,
        6856638464ULL,
        6720323584ULL,
        6584008704ULL,
        6611271680ULL,
        6693060608ULL,
        6638534656ULL,
        6952058880ULL,
        6543114240ULL,
        6924795904ULL,
        6802112512ULL,
        6747586560ULL,
        6597640192ULL,
        6774849536ULL,
        6624903168ULL,
        6870269952ULL,
        6911164416ULL,
        6938427392ULL,
        6843006976ULL,
        6733955072ULL,
        6556745728ULL,
        6679429120ULL,
        6965690368ULL,
        6761218048ULL,
        7265583104ULL,
        7320109056ULL,
        7292846080ULL,
        7142899712ULL,
        7388266496ULL,
        7156531200ULL,
        7061110784ULL,
        7074742272ULL,
        7170162688ULL,
        7047479296ULL,
        7115636736ULL,
        7279214592ULL,
        7020216320ULL,
        7183794176ULL,
        7006584832ULL,
        7361003520ULL,
        7129268224ULL,
        7197425664ULL,
        7224688640ULL,
        6979321856ULL,
        7374635008ULL,
        7251951616ULL,
        7401897984ULL,
        7088373760ULL,
        7347372032ULL,
        6992953344ULL,
        7102005248ULL,
        7306477568ULL,
        7211057152ULL,
        7033847808ULL,
        7238320128ULL,
        7333740544ULL,
        7729053696ULL,
        7769948160ULL,
        7701790720ULL,
        7510949888ULL,
        7470055424ULL,
        7538212864ULL,
        7824474112ULL,
        7551844352ULL,
        7456423936ULL,
        7742685184ULL,
        7592738816ULL,
        7415529472ULL,
        7633633280ULL,
        7442792448ULL,
        7429160960ULL,
        7620001792ULL,
        7756316672ULL,
        7524581376ULL,
        7797211136ULL,
        7497318400ULL,
        7783579648ULL,
        7647264768ULL,
        7660896256ULL,
        7606370304ULL,
        7674527744ULL,
        7565475840ULL,
        7688159232ULL,
        7715422208ULL,
        7483686912ULL,
        7810842624ULL,
        7579107328ULL,
        7838105600ULL,
        8192524288ULL,
        8097103872ULL,
        7879000064ULL,
        8219787264ULL,
        7906263040ULL,
        7988051968ULL,
        8015314944ULL,
        8178892800ULL,
        8206155776ULL,
        8001683456ULL,
        8233418752ULL,
        8165261312ULL,
        8028946432ULL,
        8274313216ULL,
        8110735360ULL,
        7974420480ULL,
        7947157504ULL,
        8069840896ULL,
        7919894528ULL,
        8042577920ULL,
        8137998336ULL,
        8260681728ULL,
        7892631552ULL,
        7960788992ULL,
        7933526016ULL,
        7851737088ULL,
        8124366848ULL,
        8151629824ULL,
        7865368576ULL,
        8247050240ULL,
        8083472384ULL,
        8056209408ULL,
        8383365120ULL,
        8328839168ULL,
        8615100416ULL,
        8506048512ULL,
        8560574464ULL,
        8642363392ULL,
        8465154048ULL,
        8396996608ULL,
        8683257856ULL,
        8424259584ULL,
        8669626368ULL,
        8710520832ULL,
        8342470656ULL,
        8478785536ULL,
        8287944704ULL,
        8519680000ULL,
        8437891072ULL,
        8301576192ULL,
        8628731904ULL,
        8356102144ULL,
        8546942976ULL,
        8696889344ULL,
        8451522560ULL,
        8574205952ULL,
        8315207680ULL,
        8533311488ULL,
        8601468928ULL,
        8655994880ULL,
        8410628096ULL,
        8369733632ULL,
        8587837440ULL,
        8492417024ULL,
        8914993152ULL,
        9105833984ULL,
        8751415296ULL,
        9078571008ULL,
        8778678272ULL,
        8887730176ULL,
        8983150592ULL,
        9119465472ULL,
        8819572736ULL,
        8846835712ULL,
        8860467200ULL,
        9010413568ULL,
        8792309760ULL,
        9051308032ULL,
        8955887616ULL,
        9092202496ULL,
        8942256128ULL,
        9133096960ULL,
        8765046784ULL,
        8805941248ULL,
        9064939520ULL,
        8996782080ULL,
        8874098688ULL,
        8724152320ULL,
        9024045056ULL,
        8901361664ULL,
        8969519104ULL,
        9146728448ULL,
        8737783808ULL,
        8833204224ULL,
        9037676544ULL,
        8928624640ULL,
        9582936064ULL,
        9187622912ULL,
        9214885888ULL,
        9337569280ULL,
        9296674816ULL,
        9569304576ULL,
        9392095232ULL,
        9228517376ULL,
        9160359936ULL,
        9283043328ULL,
        9378463744ULL,
        9323937792ULL,
        9528410112ULL,
        9419358208ULL,
        9514778624ULL,
        9555673088ULL,
        9310306304ULL,
        9446621184ULL,
        9242148864ULL,
        9487515648ULL,
        9351200768ULL,
        9405726720ULL,
        9460252672ULL,
        9473884160ULL,
        9201254400ULL,
        9501147136ULL,
        9364832256ULL,
        9173991424ULL,
        9542041600ULL,
        9432989696ULL,
        9255780352ULL,
        9269411840ULL,
        9937354752ULL,
        9869197312ULL,
        9623830528ULL,
        9637462016ULL,
        9964617728ULL,
        9801039872ULL,
        9678356480ULL,
        9828302848ULL,
        9991880704ULL,
        9719250944ULL,
        9732882432ULL,
        9746513920ULL,
        9760145408ULL,
        9841934336ULL,
        9950986240ULL,
        9882828800ULL,
        10019143680ULL,
        9691987968ULL,
        10005512192ULL,
        9651093504ULL,
        9896460288ULL,
        9787408384ULL,
        9814671360ULL,
        9910091776ULL,
        9596567552ULL,
        9923723264ULL,
        9664724992ULL,
        9855565824ULL,
        9705619456ULL,
        9773776896ULL,
        9610199040ULL,
        9978249216ULL,
        10264510464ULL,
        10046406656ULL,
        10319036416ULL,
        10073669632ULL,
        10196353024ULL,
        10100932608ULL,
        10373562368ULL,
        10128195584ULL,
        10169090048ULL,
        10223616000ULL,
        10209984512ULL,
        10141827072ULL,
        10087301120ULL,
        10359930880ULL,
        10032775168ULL,
        10414456832ULL,
        10250878976ULL,
        10155458560ULL,
        10278141952ULL,
        10455351296ULL,
        10060038144ULL,
        10332667904ULL,
        10346299392ULL,
        10305404928ULL,
        10182721536ULL,
        10114564096ULL,
        10428088320ULL,
        10400825344ULL,
        10237247488ULL,
        10387193856ULL,
        10441719808ULL,
        10291773440ULL
    };
    const char* layout = std::getenv("H40_EXPERT_LAYOUT");
    const bool use_repacked = layout != nullptr && std::string_view(layout) == "v2";
    h40::ModelIndex index;
    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        for (std::uint32_t expert = 0; expert < kExperts; ++expert) {
            const auto ordinal = static_cast<std::uint64_t>(layer) * kExperts + expert;
            const auto offset = use_repacked ? kRepackedOffsets[ordinal] : ordinal * kExpertStrideBytes;
            index.put({layer, expert}, {offset, kExpertPayloadBytes});
        }
    }
    return index;
}

h40::GptOssExpertView expert_view(std::span<const std::byte> bytes) {
    if (bytes.size() != kExpertPayloadBytes) throw std::invalid_argument("unexpected expert payload size");
    const auto* base = bytes.data();
    auto u16 = [](const std::byte* ptr, std::size_t count) {
        return std::span<const std::uint16_t>(reinterpret_cast<const std::uint16_t*>(ptr), count);
    };
    auto u8 = [](const std::byte* ptr, std::size_t count) {
        return std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(ptr), count);
    };
    return {
        kHidden,
        kIntermediate,
        u16(base + 0, kHidden),
        u8(base + 5760, kHidden * 90 * 16),
        u8(base + 4152960, kHidden * 90),
        u16(base + 4412160, kIntermediate * 2),
        u8(base + 4423680, kIntermediate * 2 * 90 * 16),
        u8(base + 12718080, kIntermediate * 2 * 90),
    };
}

void yarn_rope_tables(std::size_t seq_len, std::span<float> cos, std::span<float> sin) {
    if (cos.size() != seq_len * (kHeadDim / 2) || sin.size() != cos.size()) {
        throw std::invalid_argument("rope table shape mismatch");
    }
    constexpr double base = 150000.0;
    constexpr double factor = 32.0;
    constexpr double beta_fast = 32.0;
    constexpr double beta_slow = 1.0;
    constexpr double original_max_position_embeddings = 4096.0;
    constexpr double pi = 3.14159265358979323846264338327950288;
    const double attention_factor = 0.1 * std::log(factor) + 1.0;
    auto correction_dim = [](double rotations) {
        return (static_cast<double>(kHeadDim) * std::log(original_max_position_embeddings / (rotations * 2.0 * pi))) /
               (2.0 * std::log(base));
    };
    const double low = std::max(correction_dim(beta_fast), 0.0);
    const double high = std::min(correction_dim(beta_slow), static_cast<double>(kHeadDim - 1));
    for (std::size_t i = 0; i < kHeadDim / 2; ++i) {
        const double pos_freq = std::pow(base, static_cast<double>(i * 2) / static_cast<double>(kHeadDim));
        const double inv_extrapolate = 1.0 / pos_freq;
        const double inv_interpolate = 1.0 / (factor * pos_freq);
        const double ramp = std::clamp((static_cast<double>(i) - low) / (high - low), 0.0, 1.0);
        const double extrapolate_factor = 1.0 - ramp;
        const double inv_freq = inv_interpolate * (1.0 - extrapolate_factor) + inv_extrapolate * extrapolate_factor;
        for (std::size_t pos = 0; pos < seq_len; ++pos) {
            const double angle = static_cast<double>(pos) * inv_freq;
            cos[pos * (kHeadDim / 2) + i] = static_cast<float>(std::cos(angle) * attention_factor);
            sin[pos * (kHeadDim / 2) + i] = static_cast<float>(std::sin(angle) * attention_factor);
        }
    }
}

void apply_rope_all(std::size_t seq_len, std::span<float> q, std::span<float> k, std::span<const float> cos, std::span<const float> sin) {
    for (std::size_t pos = 0; pos < seq_len; ++pos) {
        const auto c = cos.subspan(pos * (kHeadDim / 2), kHeadDim / 2);
        const auto s = sin.subspan(pos * (kHeadDim / 2), kHeadDim / 2);
        auto q_row = q.subspan(pos * kQDim, kQDim);
        auto k_row = k.subspan(pos * kKvDim, kKvDim);
        for (std::size_t head = 0; head < kQHeads; ++head) {
            h40::apply_rope_to_head(q_row.subspan(head * kHeadDim, kHeadDim), c, s);
        }
        for (std::size_t head = 0; head < kKvHeads; ++head) {
            h40::apply_rope_to_head(k_row.subspan(head * kHeadDim, kHeadDim), c, s);
        }
    }
}

void sequence_attention(
    std::size_t seq_len,
    bool sliding,
    std::span<const float> q,
    std::span<const float> k,
    std::span<const float> v,
    std::span<const float> sinks,
    std::span<float> merged) {
    if (q.size() != seq_len * kQDim || k.size() != seq_len * kKvDim || v.size() != seq_len * kKvDim ||
        merged.size() != seq_len * kQDim) {
        throw std::invalid_argument("sequence attention shape mismatch");
    }
    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    const std::size_t group = kQHeads / kKvHeads;
    constexpr std::size_t window = 128;
    for (std::size_t pos = 0; pos < seq_len; ++pos) {
        auto out_row = merged.subspan(pos * kQDim, kQDim);
        const std::size_t min_src = (!sliding || pos + 1 <= window) ? 0 : pos + 1 - window;
        for (std::size_t qh = 0; qh < kQHeads; ++qh) {
            const std::size_t kvh = qh / group;
            const auto qv = q.subspan(pos * kQDim + qh * kHeadDim, kHeadDim);
            std::vector<float> scores(pos - min_src + 1);
            float max_score = sinks[qh];
            for (std::size_t src = min_src; src <= pos; ++src) {
                const auto kv = k.subspan(src * kKvDim + kvh * kHeadDim, kHeadDim);
                float score = 0.0F;
                for (std::size_t i = 0; i < kHeadDim; ++i) score += qv[i] * kv[i];
                score *= scale;
                scores[src - min_src] = score;
                max_score = std::max(max_score, score);
            }
            double denom = std::exp(static_cast<double>(sinks[qh] - max_score));
            for (const float score : scores) denom += std::exp(static_cast<double>(score - max_score));
            auto out = out_row.subspan(qh * kHeadDim, kHeadDim);
            std::fill(out.begin(), out.end(), 0.0F);
            for (std::size_t src = min_src; src <= pos; ++src) {
                const double prob = std::exp(static_cast<double>(scores[src - min_src] - max_score)) / denom;
                const auto vv = v.subspan(src * kKvDim + kvh * kHeadDim, kHeadDim);
                for (std::size_t i = 0; i < kHeadDim; ++i) out[i] += static_cast<float>(prob * vv[i]);
            }
        }
    }
}

void write_json(const std::filesystem::path& path, const Metrics& metrics, std::uint64_t elapsed, std::size_t input_tokens) {
    std::ofstream out(path);
    out << "{\n";
    out << "  \"schema_version\": 1,\n";
    out << "  \"status\": \"pass\",\n";
    out << "  \"mode\": \"" << (input_tokens == 1 ? "single_token_full_24_layer_h40m_decode" : "multi_token_full_24_layer_h40m_prefill_decode") << "\",\n";
    out << "  \"input_tokens\": " << input_tokens << ",\n";
    out << "  \"layers_run\": " << metrics.layers_run << ",\n";
    out << "  \"emitted_token_id\": " << metrics.token_id << ",\n";
    out << "  \"emitted_token_text\": null,\n";
    out << "  \"emitted_token_logit\": " << metrics.token_logit << ",\n";
    out << "  \"dense_flash_bytes\": " << metrics.dense_bytes << ",\n";
    out << "  \"expert_flash_bytes\": " << metrics.expert_flash_bytes << ",\n";
    out << "  \"cache_hits\": " << metrics.expert_cache_hits << ",\n";
    out << "  \"cache_misses\": " << metrics.expert_cache_misses << ",\n";
    out << "  \"cache_policy\": \"" << metrics.cache_policy << "\",\n";
    out << "  \"expert_reuse_mode\": \"" << metrics.expert_reuse_mode << "\",\n";
    out << "  \"expert_reuse_window\": " << metrics.expert_reuse_window << ",\n";
    out << "  \"expert_reuse_hits\": " << metrics.expert_reuse_hits << ",\n";
    out << "  \"expert_reuse_misses\": " << metrics.expert_reuse_misses << ",\n";
    out << "  \"peak_rss_kib\": " << metrics.peak_rss_kib << ",\n";
    out << "  \"io_overlap_enabled\": " << (metrics.io_overlap_enabled ? "true" : "false") << ",\n";
    out << "  \"prefetched_experts\": " << metrics.prefetched_experts << ",\n";
    out << "  \"prefetch_read_ns\": " << metrics.prefetch_read_ns << ",\n";
    out << "  \"prefetch_wait_ns\": " << metrics.prefetch_wait_ns << ",\n";
    out << "  \"embedding_ns\": " << metrics.embedding_ns << ",\n";
    out << "  \"dense_matvec_ns\": " << metrics.dense_matvec_ns << ",\n";
    out << "  \"attention_ns\": " << metrics.attention_ns << ",\n";
    out << "  \"moe_ns\": " << metrics.moe_ns << ",\n";
    out << "  \"lm_head_ns\": " << metrics.lm_head_ns << ",\n";
    out << "  \"dense_threads\": " << metrics.dense_threads << ",\n";
    out << "  \"lm_head_chunk_rows\": " << kLmHeadChunkRows << ",\n";
    out << "  \"elapsed_ms\": " << elapsed << "\n";
    out << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 6 && argc != 7) {
        std::cerr << "usage: minimal_decoder_probe <source_dir> <catalog.tsv> <expert_arena.bin> <token_id> <out.json> [trace.jsonl]\n";
        return 2;
    }
    const auto start = std::chrono::steady_clock::now();
    const std::filesystem::path source_dir = argv[1];
    const std::filesystem::path catalog_path = argv[2];
    const std::filesystem::path expert_arena = argv[3];
    const auto input_tokens = parse_tokens(argv[4]);
    const std::size_t seq_len = input_tokens.size();
    const std::filesystem::path out_json = argv[5];
    std::ofstream trace_file;
    std::unique_ptr<h40::JsonlTraceWriter> trace_owner;
    h40::JsonlTraceWriter* trace = nullptr;
    if (argc == 7) {
        trace_file.open(argv[6]);
        if (!trace_file) throw std::runtime_error("failed to open trace output");
        trace_owner = std::make_unique<h40::JsonlTraceWriter>(trace_file);
        trace = trace_owner.get();
    }

    Metrics metrics;
    const auto catalog = h40::H40mTensorCatalog::load_tsv(catalog_path);
    h40::FileTensorReader reader(source_dir);
    std::size_t dense_threads = kDefaultDenseThreads;
    if (const char* setting = std::getenv("H40_THREADS")) {
        dense_threads = static_cast<std::size_t>(std::stoul(setting));
    }
    if (dense_threads == 0 || dense_threads > kMaxDenseThreads) {
        throw std::invalid_argument("H40_THREADS must be in [1, 8]");
    }
    h40::ParallelBf16Matvec dense_executor(reader, kMaxDenseThreads, std::max(kQDim, kHidden));
    metrics.dense_threads = dense_threads;
    h40::FlashTensorProvider expert_provider(expert_arena);
    const auto model_index = build_expert_index();
    h40::ExpertLoader loader(model_index, expert_provider);
    h40::CachePolicy cache_policy = h40::CachePolicy::per_layer_hotset;
    if (const char* setting = std::getenv("H40_CACHE_POLICY")) {
        const std::string_view name(setting);
        if (name == "lfu_decay") {
            cache_policy = h40::CachePolicy::lfu_decay;
        } else if (name == "per_layer_hotset") {
            cache_policy = h40::CachePolicy::per_layer_hotset;
        } else if (name != "lru") {
            throw std::invalid_argument("H40_CACHE_POLICY must be lru, lfu_decay, or per_layer_hotset");
        }
    }
    metrics.cache_policy = h40::cache_policy_name(cache_policy);
    h40::ExpertCache cache(kExpertPayloadBytes * kTopK, kExpertPayloadBytes, 1048576, cache_policy);
    const char* overlap_setting = std::getenv("H40_IO_OVERLAP");
    const bool io_overlap_enabled = overlap_setting == nullptr || std::string_view(overlap_setting) != "0";
    std::vector<std::byte> prefetch_storage;
    std::unique_ptr<h40::ExpertReadPipeline> read_pipeline;
    if (io_overlap_enabled) {
        prefetch_storage.resize(kExpertPayloadBytes);
        read_pipeline = std::make_unique<h40::ExpertReadPipeline>(loader, prefetch_storage);
    }
    metrics.io_overlap_enabled = io_overlap_enabled;
    ExpertReuseMode reuse_mode = ExpertReuseMode::off;
    if (const char* setting = std::getenv("H40_EXPERT_REUSE")) {
        const std::string_view name(setting);
        if (name == "exact") {
            reuse_mode = ExpertReuseMode::exact;
        } else if (name == "approximate") {
            reuse_mode = ExpertReuseMode::approximate;
        } else if (name != "off") {
            throw std::invalid_argument("H40_EXPERT_REUSE must be off, exact, or approximate");
        }
    }
    std::size_t reuse_window = 0;
    if (reuse_mode != ExpertReuseMode::off) {
        const char* setting = std::getenv("H40_REUSE_WINDOW");
        reuse_window = setting == nullptr ? 1 : static_cast<std::size_t>(std::stoul(setting));
        if (reuse_window == 0 || reuse_window > 64) {
            throw std::invalid_argument("H40_REUSE_WINDOW must be in [1, 64]");
        }
    }
    metrics.expert_reuse_mode = reuse_mode_name(reuse_mode);
    metrics.expert_reuse_window = reuse_window;

    std::vector<float> hidden(seq_len * kHidden);
    std::vector<float> normed(seq_len * kHidden);
    std::vector<float> norm_weight(kHidden);
    std::vector<float> q(seq_len * kQDim);
    std::vector<float> k(seq_len * kKvDim);
    std::vector<float> v(seq_len * kKvDim);
    std::vector<float> q_bias(kQDim);
    std::vector<float> k_bias(kKvDim);
    std::vector<float> v_bias(kKvDim);
    std::vector<float> o_bias(kHidden);
    std::vector<float> sinks(kQHeads);
    std::vector<float> merged(seq_len * kQDim);
    std::vector<float> attn_out(seq_len * kHidden);
    std::vector<float> router_logits(seq_len * kExperts);
    std::vector<float> router_bias(kExperts);
    std::vector<float> moe_out(seq_len * kHidden);
    std::vector<float> expert_out(kHidden);
    std::vector<float> gate_up(kIntermediate * 2);
    std::vector<float> expert_hidden(kIntermediate);
    std::vector<std::uint32_t> expert_ids(kTopK);
    std::vector<float> expert_weights(kTopK);
    std::vector<float> reuse_inputs;
    std::vector<float> reuse_outputs;
    std::vector<std::size_t> reuse_tokens(kExperts);
    std::vector<bool> reuse_valid(kExperts);
    if (reuse_mode != ExpertReuseMode::off) {
        reuse_inputs.resize(kExperts * kHidden);
        reuse_outputs.resize(kExperts * kHidden);
    }

    std::vector<float> rope_cos(seq_len * (kHeadDim / 2));
    std::vector<float> rope_sin(seq_len * (kHeadDim / 2));
    yarn_rope_tables(seq_len, rope_cos, rope_sin);

    const auto embedding = must_find(catalog, "model.embed_tokens.weight");
    const auto embedding_start = std::chrono::steady_clock::now();
    for (std::size_t pos = 0; pos < seq_len; ++pos) {
        bf16_row_counted(reader, embedding, input_tokens[pos], std::span<float>(hidden).subspan(pos * kHidden, kHidden), metrics);
    }
    metrics.embedding_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - embedding_start)
            .count());

    for (std::uint32_t layer = 0; layer < kLayers; ++layer) {
        const auto prefix = std::string("model.layers.") + std::to_string(layer);
        if (trace) {
            h40::TraceEvent row;
            row.event = "decoder_layer_begin";
            row.layer = layer;
            row.has_layer = true;
            trace->emit(row);
        }
        bf16_vector_counted(reader, must_find(catalog, prefix + ".input_layernorm.weight"), norm_weight, metrics);
        for (std::size_t pos = 0; pos < seq_len; ++pos) {
            h40::rms_norm(
                std::span<const float>(hidden).subspan(pos * kHidden, kHidden),
                norm_weight,
                1.0e-5F,
                std::span<float>(normed).subspan(pos * kHidden, kHidden));
        }

        const auto q_weight = must_find(catalog, prefix + ".self_attn.q_proj.weight");
        const auto k_weight = must_find(catalog, prefix + ".self_attn.k_proj.weight");
        const auto v_weight = must_find(catalog, prefix + ".self_attn.v_proj.weight");
        for (std::size_t pos = 0; pos < seq_len; ++pos) {
            const auto row = std::span<const float>(normed).subspan(pos * kHidden, kHidden);
            bf16_matvec_counted(dense_executor, q_weight, row, std::span<float>(q).subspan(pos * kQDim, kQDim), dense_threads, metrics);
            bf16_matvec_counted(dense_executor, k_weight, row, std::span<float>(k).subspan(pos * kKvDim, kKvDim), dense_threads, metrics);
            bf16_matvec_counted(dense_executor, v_weight, row, std::span<float>(v).subspan(pos * kKvDim, kKvDim), dense_threads, metrics);
        }
        bf16_vector_counted(reader, must_find(catalog, prefix + ".self_attn.q_proj.bias"), q_bias, metrics);
        bf16_vector_counted(reader, must_find(catalog, prefix + ".self_attn.k_proj.bias"), k_bias, metrics);
        bf16_vector_counted(reader, must_find(catalog, prefix + ".self_attn.v_proj.bias"), v_bias, metrics);
        bf16_vector_counted(reader, must_find(catalog, prefix + ".self_attn.o_proj.bias"), o_bias, metrics);
        bf16_vector_counted(reader, must_find(catalog, prefix + ".self_attn.sinks"), sinks, metrics);
        for (std::size_t pos = 0; pos < seq_len; ++pos) {
            add_bias(std::span<float>(q).subspan(pos * kQDim, kQDim), q_bias);
            add_bias(std::span<float>(k).subspan(pos * kKvDim, kKvDim), k_bias);
            add_bias(std::span<float>(v).subspan(pos * kKvDim, kKvDim), v_bias);
        }
        const auto attention_start = std::chrono::steady_clock::now();
        apply_rope_all(seq_len, q, k, rope_cos, rope_sin);
        sequence_attention(seq_len, layer % 2 == 0, q, k, v, sinks, merged);
        metrics.attention_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - attention_start)
                .count());
        const auto o_weight = must_find(catalog, prefix + ".self_attn.o_proj.weight");
        for (std::size_t pos = 0; pos < seq_len; ++pos) {
            bf16_matvec_counted(
                dense_executor,
                o_weight,
                std::span<const float>(merged).subspan(pos * kQDim, kQDim),
                std::span<float>(attn_out).subspan(pos * kHidden, kHidden),
                dense_threads,
                metrics);
            add_bias(std::span<float>(attn_out).subspan(pos * kHidden, kHidden), o_bias);
        }
        add_inplace(hidden, attn_out);
        if (trace) {
            h40::TraceEvent row;
            row.event = "attention_end";
            row.layer = layer;
            row.has_layer = true;
            row.bytes = q.size() * sizeof(float) + k.size() * sizeof(float) + v.size() * sizeof(float);
            row.has_bytes = true;
            trace->emit(row);
        }

        bf16_vector_counted(reader, must_find(catalog, prefix + ".post_attention_layernorm.weight"), norm_weight, metrics);
        for (std::size_t pos = 0; pos < seq_len; ++pos) {
            h40::rms_norm(
                std::span<const float>(hidden).subspan(pos * kHidden, kHidden),
                norm_weight,
                1.0e-5F,
                std::span<float>(normed).subspan(pos * kHidden, kHidden));
        }
        const auto router_weight = must_find(catalog, prefix + ".mlp.router.weight");
        for (std::size_t pos = 0; pos < seq_len; ++pos) {
            bf16_matvec_counted(
                dense_executor,
                router_weight,
                std::span<const float>(normed).subspan(pos * kHidden, kHidden),
                std::span<float>(router_logits).subspan(pos * kExperts, kExperts),
                dense_threads,
                metrics);
        }
        bf16_vector_counted(reader, must_find(catalog, prefix + ".mlp.router.bias"), router_bias, metrics);
        for (std::size_t pos = 0; pos < seq_len; ++pos) {
            add_bias(std::span<float>(router_logits).subspan(pos * kExperts, kExperts), router_bias);
        }

        const auto moe_start = std::chrono::steady_clock::now();
        std::fill(reuse_valid.begin(), reuse_valid.end(), false);
        h40::run_moe_layer_streaming(
            {layer, seq_len, kExperts, kTopK, kHidden},
            router_logits,
            cache,
            loader,
            moe_out,
            {expert_ids, expert_weights, expert_out},
            [&](std::size_t token, std::uint32_t expert, std::span<const std::byte> packed, std::span<float> out) {
                const auto input = std::span<const float>(normed).subspan(token * kHidden, kHidden);
                if (reuse_mode != ExpertReuseMode::off) {
                    const auto cached_input = std::span<const float>(reuse_inputs).subspan(expert * kHidden, kHidden);
                    const auto cached_output = std::span<const float>(reuse_outputs).subspan(expert * kHidden, kHidden);
                    const bool in_window = reuse_valid[expert] && token > reuse_tokens[expert] &&
                                           token - reuse_tokens[expert] <= reuse_window;
                    const bool exact_match = in_window && std::equal(input.begin(), input.end(), cached_input.begin());
                    const bool reuse = in_window &&
                                       (reuse_mode == ExpertReuseMode::approximate || exact_match);
                    if (reuse) {
                        std::copy(cached_output.begin(), cached_output.end(), out.begin());
                        reuse_tokens[expert] = token;
                        ++metrics.expert_reuse_hits;
                        return;
                    }
                    ++metrics.expert_reuse_misses;
                }
                h40::run_gptoss_expert(
                    expert_view(packed),
                    input,
                    out,
                    {gate_up, expert_hidden});
                if (reuse_mode != ExpertReuseMode::off) {
                    std::copy(input.begin(), input.end(), reuse_inputs.begin() + expert * kHidden);
                    std::copy(out.begin(), out.end(), reuse_outputs.begin() + expert * kHidden);
                    reuse_tokens[expert] = token;
                    reuse_valid[expert] = true;
                }
            },
            trace,
            false,
            read_pipeline.get());
        metrics.moe_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - moe_start)
                .count());
        add_inplace(hidden, moe_out);
        ++metrics.layers_run;
        if (trace) {
            h40::TraceEvent row;
            row.event = "decoder_layer_end";
            row.layer = layer;
            row.has_layer = true;
            trace->emit(row);
        }
    }

    bf16_vector_counted(reader, must_find(catalog, "model.norm.weight"), norm_weight, metrics);
    const auto last_hidden = std::span<const float>(hidden).subspan((seq_len - 1) * kHidden, kHidden);
    auto last_normed = std::span<float>(normed).subspan((seq_len - 1) * kHidden, kHidden);
    h40::rms_norm(last_hidden, norm_weight, 1.0e-5F, last_normed);

    const auto lm_head = must_find(catalog, "lm_head.weight");
    std::vector<float> logits(kLmHeadChunkRows);
    const auto lm_head_start = std::chrono::steady_clock::now();
    for (std::size_t row = 0; row < kVocab; row += kLmHeadChunkRows) {
        const auto rows = std::min(kLmHeadChunkRows, kVocab - row);
        auto chunk = std::span<float>(logits).first(rows);
        dense_executor.matvec_rows(lm_head, row, last_normed, chunk, dense_threads);
        metrics.dense_bytes += rows * kHidden * sizeof(std::uint16_t);
        for (std::size_t i = 0; i < rows; ++i) {
            const float value = chunk[i];
            const auto id = static_cast<std::uint32_t>(row + i);
            if (value > metrics.token_logit || (value == metrics.token_logit && id < metrics.token_id)) {
                metrics.token_logit = value;
                metrics.token_id = id;
            }
        }
    }
    metrics.lm_head_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - lm_head_start)
            .count());

    const auto stats = cache.stats();
    metrics.expert_cache_hits = stats.hits;
    metrics.expert_cache_misses = stats.misses;
    metrics.expert_flash_bytes = stats.bytes_loaded;
    if (read_pipeline) {
        const auto prefetch_stats = read_pipeline->stats();
        metrics.prefetched_experts = prefetch_stats.completed;
        metrics.prefetch_read_ns = prefetch_stats.read_nanoseconds;
        metrics.prefetch_wait_ns = prefetch_stats.wait_nanoseconds;
    }
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        metrics.peak_rss_kib = static_cast<std::uint64_t>(usage.ru_maxrss);
    }
    write_json(out_json, metrics, elapsed_ms(start), seq_len);
    if (trace) {
        h40::TraceEvent row;
        row.event = "streamed_lm_head_argmax";
        row.token = metrics.token_id;
        row.has_token = true;
        row.bytes = kVocab * kHidden * sizeof(std::uint16_t);
        row.has_bytes = true;
        trace->emit(row);
    }
    std::cout << "emitted_token_id=" << metrics.token_id << "\n";
    std::cout << "emitted_token_logit=" << metrics.token_logit << "\n";
    return 0;
}
