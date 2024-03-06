/* SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Copyright:
 *   2024      Guation <guation@guation.cn>
 */

#define SIMDE_TEST_X86_AVX512_INSN bs

#include <test/x86/avx512/test-avx512.h>
#include <simde/x86/avx512/scatter.h>
#include <simde/x86/avx512/gather.h>

static int32_t i32_buffer[2048];

static int
test_simde_mm512_i32scatter_epi32 (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t vindex[16];
    const int32_t a[16];
  } test_vec[] = {
    { {  INT32_C(         253),  INT32_C(         183),  INT32_C(         102),  INT32_C(          84),  INT32_C(         182),  INT32_C(         112),  INT32_C(          89),  INT32_C(         199),
         INT32_C(         168),  INT32_C(          55),  INT32_C(          94),  INT32_C(         125),  INT32_C(         239),  INT32_C(         128),  INT32_C(         194),  INT32_C(           5) },
      { -INT32_C(  1691445770), -INT32_C(  1911279065),  INT32_C(    29953023), -INT32_C(  1940439154), -INT32_C(  1377780234), -INT32_C(   713803475),  INT32_C(  1599328880),  INT32_C(  1231295827),
        -INT32_C(   756706390), -INT32_C(   983500347), -INT32_C(  1815729916),  INT32_C(  1595874664),  INT32_C(   118292442), -INT32_C(  1461878728), -INT32_C(  1526255534), -INT32_C(   252744891) } },
    { {  INT32_C(         254),  INT32_C(         212),  INT32_C(         194),  INT32_C(         196),  INT32_C(         205),  INT32_C(          34),  INT32_C(         137),  INT32_C(         209),
         INT32_C(          75),  INT32_C(          79),  INT32_C(         100),  INT32_C(         179),  INT32_C(         109),  INT32_C(         131),  INT32_C(          18),  INT32_C(          71) },
      { -INT32_C(  1169219966), -INT32_C(   245224290), -INT32_C(  1583978148), -INT32_C(   745437739),  INT32_C(   647451481), -INT32_C(  1057480587), -INT32_C(   579576720),  INT32_C(  1646560992),
         INT32_C(  1126003621), -INT32_C(    80445537), -INT32_C(  1096954903), -INT32_C(  1433326000), -INT32_C(   137352830), -INT32_C(  1179072439),  INT32_C(    93727781),  INT32_C(  1466416050) } },
    { {  INT32_C(          46),  INT32_C(         132),  INT32_C(         155),  INT32_C(         206),  INT32_C(           4),  INT32_C(         207),  INT32_C(         201),  INT32_C(         237),
         INT32_C(         154),  INT32_C(         102),  INT32_C(         171),  INT32_C(         235),  INT32_C(         149),  INT32_C(          60),  INT32_C(          23),  INT32_C(         101) },
      {  INT32_C(   783224421), -INT32_C(   229414714), -INT32_C(  1163568897),  INT32_C(  1156119743),  INT32_C(  1732818583), -INT32_C(   436128384),  INT32_C(   477883616), -INT32_C(   410873215),
         INT32_C(  1729442209), -INT32_C(  1738905447),  INT32_C(  2136145856), -INT32_C(  1815921669),  INT32_C(  1912212465),  INT32_C(   542702400),  INT32_C(  1765659624),  INT32_C(   122732390) } },
    { {  INT32_C(         240),  INT32_C(         101),  INT32_C(         110),  INT32_C(         137),  INT32_C(         205),  INT32_C(         200),  INT32_C(          34),  INT32_C(         141),
         INT32_C(         199),  INT32_C(         116),  INT32_C(          13),  INT32_C(         195),  INT32_C(         175),  INT32_C(         208),  INT32_C(          86),  INT32_C(         161) },
      {  INT32_C(   470962396),  INT32_C(   859662923), -INT32_C(  1533183426),  INT32_C(   699133241),  INT32_C(   548542803), -INT32_C(  1448160030), -INT32_C(   127091895),  INT32_C(  1738130059),
         INT32_C(  1585753106),  INT32_C(  1418838294),  INT32_C(  1962421819),  INT32_C(  1872601884), -INT32_C(  1617997891),  INT32_C(  1833450788), -INT32_C(  2073709064), -INT32_C(  1964245129) } },
    { {  INT32_C(         171),  INT32_C(         111),  INT32_C(         232),  INT32_C(         193),  INT32_C(          48),  INT32_C(         121),  INT32_C(          22),  INT32_C(         107),
         INT32_C(         168),  INT32_C(          14),  INT32_C(         223),  INT32_C(         196),  INT32_C(         178),  INT32_C(         124),  INT32_C(          51),  INT32_C(         204) },
      {  INT32_C(    15732418),  INT32_C(   200826198), -INT32_C(  1048347454), -INT32_C(   680784536),  INT32_C(  1846030069),  INT32_C(  1377203012), -INT32_C(   821765549), -INT32_C(   795118835),
        -INT32_C(   657421439), -INT32_C(  1411135256), -INT32_C(  1385404859),  INT32_C(  1753536883),  INT32_C(  1272417287),  INT32_C(  1402858752), -INT32_C(   719150392), -INT32_C(  1750745579) } },
    { {  INT32_C(          73),  INT32_C(         117),  INT32_C(         111),  INT32_C(          50),  INT32_C(          62),  INT32_C(          82),  INT32_C(         221),  INT32_C(         131),
         INT32_C(         185),  INT32_C(          48),  INT32_C(          44),  INT32_C(          34),  INT32_C(         180),  INT32_C(         149),  INT32_C(          41),  INT32_C(          65) },
      {  INT32_C(  1497461868), -INT32_C(  1272867822),  INT32_C(  1959393206),  INT32_C(   314466460),  INT32_C(   575729871),  INT32_C(   383505357), -INT32_C(  1220999421),  INT32_C(   150495900),
        -INT32_C(   379438633), -INT32_C(  2070051891),  INT32_C(   402155130), -INT32_C(  1775651129),  INT32_C(  1941469606),  INT32_C(  1334481740),  INT32_C(   923190171), -INT32_C(    62849243) } },
    { {  INT32_C(          56),  INT32_C(         162),  INT32_C(         229),  INT32_C(           6),  INT32_C(          37),  INT32_C(         130),  INT32_C(         138),  INT32_C(         160),
         INT32_C(         233),  INT32_C(         183),  INT32_C(         176),  INT32_C(          57),  INT32_C(         224),  INT32_C(          70),  INT32_C(         223),  INT32_C(          89) },
      { -INT32_C(  1834658817), -INT32_C(  1607600931), -INT32_C(   104503814),  INT32_C(  1194508965),  INT32_C(   711735463),  INT32_C(  1142099138), -INT32_C(  1552038973),  INT32_C(   150756617),
        -INT32_C(  1919180368), -INT32_C(  1876047723), -INT32_C(   762711507),  INT32_C(  1545190324), -INT32_C(  1249474829),  INT32_C(  1459263890),  INT32_C(  1710847836), -INT32_C(  2073102636) } },
    { {  INT32_C(         151),  INT32_C(           9),  INT32_C(          17),  INT32_C(          45),  INT32_C(         209),  INT32_C(          62),  INT32_C(         189),  INT32_C(         255),
         INT32_C(          49),  INT32_C(          70),  INT32_C(         229),  INT32_C(           2),  INT32_C(         235),  INT32_C(          65),  INT32_C(         245),  INT32_C(         113) },
      {  INT32_C(  1610853319),  INT32_C(   482105765),  INT32_C(  1240539731), -INT32_C(  1713343088), -INT32_C(   982905465),  INT32_C(   301361610),  INT32_C(   638835771), -INT32_C(   443086819),
         INT32_C(  1480956595),  INT32_C(  1198850804), -INT32_C(  1265605084),  INT32_C(  1649242331),  INT32_C(  1210627966),  INT32_C(  1549344289),  INT32_C(   411200762),  INT32_C(   687675765) } },
  };
  for (size_t i = 0 ; i < (sizeof(i32_buffer) / sizeof(i32_buffer[0])) ; i++) { i32_buffer[i] = HEDLEY_STATIC_CAST(int32_t, i); }

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i vindex = simde_mm512_loadu_epi32(test_vec[i].vindex);
    simde__m512i a = simde_mm512_loadu_epi32(test_vec[i].a);
    switch (i)
    {
      case 0:
      case 1:
      case 2:
      case 3:
        simde_mm512_i32scatter_epi32(HEDLEY_STATIC_CAST(void*, i32_buffer), vindex, a, 4);
        simde_test_x86_assert_equal_i32x16(simde_mm512_i32gather_epi32(vindex, HEDLEY_STATIC_CAST(const void*, i32_buffer), 4), a);
        break;
      case 4:
      case 5:
      case 6:
      case 7:
        simde_mm512_i32scatter_epi32(HEDLEY_STATIC_CAST(void*, i32_buffer), vindex, a, 8);
        simde_test_x86_assert_equal_i32x16(simde_mm512_i32gather_epi32(vindex, HEDLEY_STATIC_CAST(const void*, i32_buffer), 8), a);
        break;
      default:
        return 1;
    }
  }
  return 0;
#else
  fputc('\n', stdout);

  (void)i32_buffer;

  for (int i = 0; i < 8; i++) {
    int32_t e[16];
    for (size_t i = 0; i < (sizeof(e) / sizeof(e[0])); i++) {
      size_t j;
      renew:
      e[i] = HEDLEY_STATIC_CAST(int32_t, (simde_test_codegen_random_u8()));
      for (j = 0; j < i; j++) if (e[j] == e[i]) goto renew;
    }
    simde__m512i vindex = simde_mm512_loadu_epi32(e);
    simde__m512i a = simde_test_x86_random_i32x16();

    simde_test_x86_write_i32x16(2, vindex, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i32x16(2, a, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_mask_i32scatter_epi32 (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const simde__mmask16 k;
    const int32_t vindex[16];
    const int32_t a[16];
  } test_vec[] = {
    { UINT16_C(11938),
      {  INT32_C(         147),  INT32_C(          28),  INT32_C(          39),  INT32_C(         106),  INT32_C(          48),  INT32_C(          36),  INT32_C(          61),  INT32_C(         223),
         INT32_C(         181),  INT32_C(          81),  INT32_C(         228),  INT32_C(          96),  INT32_C(          16),  INT32_C(         177),  INT32_C(         221),  INT32_C(          59) },
      {  INT32_C(  1811123971), -INT32_C(   648053051),  INT32_C(  1690916530),  INT32_C(   368572665),  INT32_C(  1145397535), -INT32_C(   268884834), -INT32_C(  1174382072),  INT32_C(   989138999),
         INT32_C(  1705371808),  INT32_C(   423495015),  INT32_C(  1417480283),  INT32_C(   527004672),  INT32_C(  1952689877), -INT32_C(   614244910), -INT32_C(   292199497),  INT32_C(   338266740) } },
    { UINT16_C(52850),
      {  INT32_C(         121),  INT32_C(         217),  INT32_C(         211),  INT32_C(         184),  INT32_C(         243),  INT32_C(          47),  INT32_C(         192),  INT32_C(         112),
         INT32_C(         131),  INT32_C(         229),  INT32_C(         237),  INT32_C(         223),  INT32_C(         186),  INT32_C(         155),  INT32_C(          67),  INT32_C(          46) },
      {  INT32_C(  1234346094),  INT32_C(  1172239703),  INT32_C(  2104387689), -INT32_C(  1258865190),  INT32_C(  1067953936),  INT32_C(   801314671), -INT32_C(  1223773956), -INT32_C(  1176153781),
         INT32_C(  1208121329), -INT32_C(   695279507),  INT32_C(   592772168),  INT32_C(  1255623481),  INT32_C(  1770618618), -INT32_C(  1835512683),  INT32_C(  1212786428), -INT32_C(   385798408) } },
    { UINT16_C( 1190),
      {  INT32_C(          50),  INT32_C(          19),  INT32_C(         228),  INT32_C(         192),  INT32_C(         233),  INT32_C(          45),  INT32_C(         188),  INT32_C(          61),
         INT32_C(          80),  INT32_C(         246),  INT32_C(         136),  INT32_C(          39),  INT32_C(          64),  INT32_C(         130),  INT32_C(         165),  INT32_C(         201) },
      { -INT32_C(  2095695125),  INT32_C(   355078860),  INT32_C(  1547969114),  INT32_C(  1029761547), -INT32_C(   419609347), -INT32_C(  1054557583), -INT32_C(   253186896),  INT32_C(   448367919),
        -INT32_C(  1801531448),  INT32_C(  1017825506), -INT32_C(   174526742), -INT32_C(   718014248), -INT32_C(  1363464131), -INT32_C(  1703878678), -INT32_C(  1148561268), -INT32_C(  1361754906) } },
    { UINT16_C(29459),
      {  INT32_C(          66),  INT32_C(         245),  INT32_C(          59),  INT32_C(         236),  INT32_C(          50),  INT32_C(          37),  INT32_C(         218),  INT32_C(         202),
         INT32_C(          27),  INT32_C(         178),  INT32_C(         195),  INT32_C(          78),  INT32_C(         135),  INT32_C(           0),  INT32_C(         126),  INT32_C(          67) },
      {  INT32_C(   505571758), -INT32_C(  1904759293),  INT32_C(  1070751081),  INT32_C(  1303569675),  INT32_C(   221966043),  INT32_C(   785847315),  INT32_C(  1316788935),  INT32_C(  1234303898),
         INT32_C(  1734849636), -INT32_C(   856301982),  INT32_C(  1175177019), -INT32_C(  2003518035), -INT32_C(  1080701269), -INT32_C(  1444057886), -INT32_C(  1577555449), -INT32_C(   907376283) } },
    { UINT16_C(21053),
      {  INT32_C(          48),  INT32_C(         160),  INT32_C(          38),  INT32_C(         108),  INT32_C(         237),  INT32_C(         119),  INT32_C(         178),  INT32_C(         154),
         INT32_C(          52),  INT32_C(          70),  INT32_C(          34),  INT32_C(         224),  INT32_C(          20),  INT32_C(         183),  INT32_C(         159),  INT32_C(         247) },
      {  INT32_C(   731941924),  INT32_C(  1540135158),  INT32_C(  1596241698),  INT32_C(   973034761),  INT32_C(  1755671419),  INT32_C(   386095330), -INT32_C(  1275648610),  INT32_C(    27956957),
         INT32_C(   422332962),  INT32_C(    91551971), -INT32_C(  1201366609),  INT32_C(  1777493230), -INT32_C(  1294886961), -INT32_C(  1899375377), -INT32_C(   683556614),  INT32_C(  2027481942) } },
    { UINT16_C( 1077),
      {  INT32_C(         145),  INT32_C(          24),  INT32_C(         252),  INT32_C(           6),  INT32_C(          29),  INT32_C(         172),  INT32_C(         159),  INT32_C(         130),
         INT32_C(         100),  INT32_C(         141),  INT32_C(         230),  INT32_C(          86),  INT32_C(         246),  INT32_C(         181),  INT32_C(         238),  INT32_C(         199) },
      {  INT32_C(   815586663), -INT32_C(  1393518997),  INT32_C(  1167541869),  INT32_C(  1347014079),  INT32_C(    55985637),  INT32_C(  1451619825),  INT32_C(  2024565634), -INT32_C(  2009032160),
        -INT32_C(   474424200), -INT32_C(   544167566), -INT32_C(  1373362193), -INT32_C(   620794380), -INT32_C(  1528998478), -INT32_C(   839228853), -INT32_C(   314136883), -INT32_C(  1183480255) } },
    { UINT16_C(11874),
      {  INT32_C(         156),  INT32_C(         212),  INT32_C(         215),  INT32_C(          44),  INT32_C(         179),  INT32_C(         198),  INT32_C(          84),  INT32_C(         117),
         INT32_C(          72),  INT32_C(          68),  INT32_C(         116),  INT32_C(          34),  INT32_C(         246),  INT32_C(         201),  INT32_C(         255),  INT32_C(         154) },
      { -INT32_C(   493592300),  INT32_C(   472398638),  INT32_C(   898739836), -INT32_C(   456016112),  INT32_C(  1570242198), -INT32_C(  1697485230), -INT32_C(  1463990606),  INT32_C(   608418831),
         INT32_C(  1275516702), -INT32_C(  1905775086), -INT32_C(   322700580),  INT32_C(  1355847097), -INT32_C(   441620589), -INT32_C(  2021687339), -INT32_C(   735101755),  INT32_C(   402158329) } },
    { UINT16_C(65098),
      {  INT32_C(          99),  INT32_C(          92),  INT32_C(          44),  INT32_C(         204),  INT32_C(         235),  INT32_C(           8),  INT32_C(         198),  INT32_C(         174),
         INT32_C(         244),  INT32_C(         127),  INT32_C(          67),  INT32_C(         196),  INT32_C(         207),  INT32_C(         214),  INT32_C(          43),  INT32_C(         124) },
      {  INT32_C(  1006305467), -INT32_C(  1216888697), -INT32_C(  1926664043), -INT32_C(   343116921), -INT32_C(  1145587504), -INT32_C(  1251377727), -INT32_C(   864375300),  INT32_C(  1061725571),
         INT32_C(   762987686), -INT32_C(  1713049340), -INT32_C(   383316382),  INT32_C(  1372894081),  INT32_C(   739019627),  INT32_C(    98727432), -INT32_C(  1496228829), -INT32_C(  1478158079) } },
  };
  for (size_t i = 0 ; i < (sizeof(i32_buffer) / sizeof(i32_buffer[0])) ; i++) { i32_buffer[i] = HEDLEY_STATIC_CAST(int32_t, i); }

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i vindex = simde_mm512_loadu_epi32(test_vec[i].vindex);
    simde__m512i a = simde_mm512_loadu_epi32(test_vec[i].a);
    simde__m512i k;
    switch (i)
    {
      case 0:
      case 1:
      case 2:
      case 3:
        k = simde_mm512_i32gather_epi32(vindex, HEDLEY_STATIC_CAST(const void*, i32_buffer), 4);
        simde_mm512_mask_i32scatter_epi32(HEDLEY_STATIC_CAST(void*, i32_buffer), test_vec[i].k, vindex, a, 4);
        simde_test_x86_assert_equal_i32x16(simde_mm512_i32gather_epi32(vindex, HEDLEY_STATIC_CAST(const void*, i32_buffer), 4), simde_mm512_mask_mov_epi32(k, test_vec[i].k, a));
        break;
      case 4:
      case 5:
      case 6:
      case 7:
        k = simde_mm512_i32gather_epi32(vindex, HEDLEY_STATIC_CAST(const void*, i32_buffer), 8);
        simde_mm512_mask_i32scatter_epi32(HEDLEY_STATIC_CAST(void*, i32_buffer), test_vec[i].k, vindex, a, 8);
        simde_test_x86_assert_equal_i32x16(simde_mm512_i32gather_epi32(vindex, HEDLEY_STATIC_CAST(const void*, i32_buffer), 8), simde_mm512_mask_mov_epi32(k, test_vec[i].k, a));
        break;
      default:
        return 1;
    }
  }
  return 0;
#else
  fputc('\n', stdout);

  (void)i32_buffer;

  for (int i = 0; i < 8; i++) {
    simde__mmask16 k = simde_test_x86_random_mmask16();
    int32_t e[16];
    for (size_t i = 0; i < (sizeof(e) / sizeof(e[0])); i++) {
      size_t j;
      renew:
      e[i] = HEDLEY_STATIC_CAST(int32_t, (simde_test_codegen_random_u8()));
      for (j = 0; j < i; j++) if (e[j] == e[i]) goto renew;
    }
    simde__m512i vindex = simde_mm512_loadu_epi32(e);
    simde__m512i a = simde_test_x86_random_i32x16();

    simde_test_x86_write_mmask16(2, k, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i32x16(2, vindex, SIMDE_TEST_VEC_POS_MIDDLE);
    simde_test_x86_write_i32x16(2, a, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static simde_float32 f32_buffer[2048];

static int
test_simde_mm512_i32scatter_ps (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int32_t vindex[16];
    const simde_float32 a[16];
  } test_vec[] = {
    { {  INT32_C(         139),  INT32_C(         144),  INT32_C(         234),  INT32_C(         243),  INT32_C(          96),  INT32_C(         150),  INT32_C(         248),  INT32_C(         149),
         INT32_C(         246),  INT32_C(          58),  INT32_C(         227),  INT32_C(         156),  INT32_C(          17),  INT32_C(         115),  INT32_C(          93),  INT32_C(          43) },
      { SIMDE_FLOAT32_C(   876.99), SIMDE_FLOAT32_C(  -158.58), SIMDE_FLOAT32_C(   923.12), SIMDE_FLOAT32_C(   633.13),
        SIMDE_FLOAT32_C(   856.73), SIMDE_FLOAT32_C(  -571.42), SIMDE_FLOAT32_C(   310.48), SIMDE_FLOAT32_C(  -584.79),
        SIMDE_FLOAT32_C(  -812.08), SIMDE_FLOAT32_C(  -915.57), SIMDE_FLOAT32_C(  -799.11), SIMDE_FLOAT32_C(  -825.21),
        SIMDE_FLOAT32_C(  -299.12), SIMDE_FLOAT32_C(   -23.63), SIMDE_FLOAT32_C(    20.03), SIMDE_FLOAT32_C(  -777.94) } },
    { {  INT32_C(         133),  INT32_C(          80),  INT32_C(         154),  INT32_C(         229),  INT32_C(         230),  INT32_C(         146),  INT32_C(         123),  INT32_C(         220),
         INT32_C(         204),  INT32_C(          94),  INT32_C(         120),  INT32_C(         222),  INT32_C(         209),  INT32_C(         214),  INT32_C(           9),  INT32_C(         138) },
      { SIMDE_FLOAT32_C(   680.48), SIMDE_FLOAT32_C(   164.41), SIMDE_FLOAT32_C(  -392.35), SIMDE_FLOAT32_C(   537.22),
        SIMDE_FLOAT32_C(   592.99), SIMDE_FLOAT32_C(   918.13), SIMDE_FLOAT32_C(   952.43), SIMDE_FLOAT32_C(   780.91),
        SIMDE_FLOAT32_C(  -997.44), SIMDE_FLOAT32_C(  -846.69), SIMDE_FLOAT32_C(   955.70), SIMDE_FLOAT32_C(  -296.56),
        SIMDE_FLOAT32_C(   129.68), SIMDE_FLOAT32_C(   -24.27), SIMDE_FLOAT32_C(   -74.50), SIMDE_FLOAT32_C(   182.12) } },
    { {  INT32_C(         106),  INT32_C(         137),  INT32_C(         108),  INT32_C(          80),  INT32_C(          27),  INT32_C(         231),  INT32_C(          44),  INT32_C(         232),
         INT32_C(          70),  INT32_C(         165),  INT32_C(         198),  INT32_C(          23),  INT32_C(         123),  INT32_C(         207),  INT32_C(         162),  INT32_C(         131) },
      { SIMDE_FLOAT32_C(  -257.34), SIMDE_FLOAT32_C(   484.95), SIMDE_FLOAT32_C(   358.62), SIMDE_FLOAT32_C(  -664.35),
        SIMDE_FLOAT32_C(   403.08), SIMDE_FLOAT32_C(   311.04), SIMDE_FLOAT32_C(  -883.44), SIMDE_FLOAT32_C(   405.65),
        SIMDE_FLOAT32_C(   464.36), SIMDE_FLOAT32_C(  -927.74), SIMDE_FLOAT32_C(  -890.91), SIMDE_FLOAT32_C(  -405.96),
        SIMDE_FLOAT32_C(    47.99), SIMDE_FLOAT32_C(    34.59), SIMDE_FLOAT32_C(   776.16), SIMDE_FLOAT32_C(   -96.65) } },
    { {  INT32_C(         222),  INT32_C(         183),  INT32_C(         134),  INT32_C(         250),  INT32_C(         158),  INT32_C(         179),  INT32_C(         226),  INT32_C(         228),
         INT32_C(          88),  INT32_C(         168),  INT32_C(         252),  INT32_C(         211),  INT32_C(         119),  INT32_C(          86),  INT32_C(         207),  INT32_C(          87) },
      { SIMDE_FLOAT32_C(   264.92), SIMDE_FLOAT32_C(   836.04), SIMDE_FLOAT32_C(   413.75), SIMDE_FLOAT32_C(  -424.04),
        SIMDE_FLOAT32_C(   952.60), SIMDE_FLOAT32_C(  -180.61), SIMDE_FLOAT32_C(  -959.68), SIMDE_FLOAT32_C(  -975.13),
        SIMDE_FLOAT32_C(   -71.52), SIMDE_FLOAT32_C(  -365.64), SIMDE_FLOAT32_C(    72.86), SIMDE_FLOAT32_C(   963.07),
        SIMDE_FLOAT32_C(  -589.48), SIMDE_FLOAT32_C(   976.20), SIMDE_FLOAT32_C(  -707.48), SIMDE_FLOAT32_C(  -360.72) } },
    { {  INT32_C(          21),  INT32_C(         247),  INT32_C(         159),  INT32_C(         200),  INT32_C(         217),  INT32_C(         131),  INT32_C(          32),  INT32_C(         129),
         INT32_C(         127),  INT32_C(         243),  INT32_C(         248),  INT32_C(          29),  INT32_C(          73),  INT32_C(         117),  INT32_C(          12),  INT32_C(         162) },
      { SIMDE_FLOAT32_C(   -76.11), SIMDE_FLOAT32_C(  -120.15), SIMDE_FLOAT32_C(  -203.38), SIMDE_FLOAT32_C(   743.28),
        SIMDE_FLOAT32_C(   -79.83), SIMDE_FLOAT32_C(  -178.52), SIMDE_FLOAT32_C(  -328.23), SIMDE_FLOAT32_C(   554.53),
        SIMDE_FLOAT32_C(   894.34), SIMDE_FLOAT32_C(  -365.16), SIMDE_FLOAT32_C(   965.05), SIMDE_FLOAT32_C(   870.54),
        SIMDE_FLOAT32_C(   -72.63), SIMDE_FLOAT32_C(  -395.67), SIMDE_FLOAT32_C(  -667.83), SIMDE_FLOAT32_C(   655.19) } },
    { {  INT32_C(         169),  INT32_C(         202),  INT32_C(         216),  INT32_C(          45),  INT32_C(         234),  INT32_C(          90),  INT32_C(         172),  INT32_C(         221),
         INT32_C(          82),  INT32_C(          38),  INT32_C(          26),  INT32_C(          63),  INT32_C(          51),  INT32_C(         189),  INT32_C(         251),  INT32_C(         185) },
      { SIMDE_FLOAT32_C(  -868.43), SIMDE_FLOAT32_C(  -840.09), SIMDE_FLOAT32_C(  -240.78), SIMDE_FLOAT32_C(   -46.95),
        SIMDE_FLOAT32_C(  -168.33), SIMDE_FLOAT32_C(  -686.25), SIMDE_FLOAT32_C(  -152.61), SIMDE_FLOAT32_C(   466.52),
        SIMDE_FLOAT32_C(  -721.20), SIMDE_FLOAT32_C(  -282.07), SIMDE_FLOAT32_C(  -606.12), SIMDE_FLOAT32_C(  -116.87),
        SIMDE_FLOAT32_C(    50.10), SIMDE_FLOAT32_C(  -950.93), SIMDE_FLOAT32_C(    67.14), SIMDE_FLOAT32_C(  -513.97) } },
    { {  INT32_C(         240),  INT32_C(          38),  INT32_C(          67),  INT32_C(          74),  INT32_C(         210),  INT32_C(          33),  INT32_C(         157),  INT32_C(         156),
         INT32_C(          71),  INT32_C(         183),  INT32_C(         219),  INT32_C(         122),  INT32_C(         116),  INT32_C(         215),  INT32_C(          51),  INT32_C(         173) },
      { SIMDE_FLOAT32_C(   111.45), SIMDE_FLOAT32_C(  -201.16), SIMDE_FLOAT32_C(   643.48), SIMDE_FLOAT32_C(   943.12),
        SIMDE_FLOAT32_C(   112.59), SIMDE_FLOAT32_C(  -509.13), SIMDE_FLOAT32_C(   409.64), SIMDE_FLOAT32_C(   391.39),
        SIMDE_FLOAT32_C(   208.81), SIMDE_FLOAT32_C(   803.52), SIMDE_FLOAT32_C(  -725.48), SIMDE_FLOAT32_C(  -741.10),
        SIMDE_FLOAT32_C(   852.60), SIMDE_FLOAT32_C(   341.66), SIMDE_FLOAT32_C(  -255.07), SIMDE_FLOAT32_C(   127.22) } },
    { {  INT32_C(         156),  INT32_C(          66),  INT32_C(         120),  INT32_C(         111),  INT32_C(          99),  INT32_C(          21),  INT32_C(          11),  INT32_C(         170),
         INT32_C(         205),  INT32_C(         231),  INT32_C(          37),  INT32_C(          65),  INT32_C(         190),  INT32_C(          88),  INT32_C(         238),  INT32_C(         177) },
      { SIMDE_FLOAT32_C(  -418.89), SIMDE_FLOAT32_C(   226.89), SIMDE_FLOAT32_C(  -439.05), SIMDE_FLOAT32_C(   693.70),
        SIMDE_FLOAT32_C(   717.76), SIMDE_FLOAT32_C(   970.59), SIMDE_FLOAT32_C(    85.09), SIMDE_FLOAT32_C(   -73.43),
        SIMDE_FLOAT32_C(   774.11), SIMDE_FLOAT32_C(   359.61), SIMDE_FLOAT32_C(   185.47), SIMDE_FLOAT32_C(   626.71),
        SIMDE_FLOAT32_C(  -298.73), SIMDE_FLOAT32_C(   930.40), SIMDE_FLOAT32_C(  -246.08), SIMDE_FLOAT32_C(  -380.59) } },
  };
  for (size_t i = 0 ; i < (sizeof(f32_buffer) / sizeof(f32_buffer[0])) ; i++) { f32_buffer[i] = HEDLEY_STATIC_CAST(simde_float32, i); }

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i vindex = simde_mm512_loadu_epi32(test_vec[i].vindex);
    simde__m512 a = simde_mm512_loadu_ps(test_vec[i].a);
    switch (i)
    {
      case 0:
      case 1:
      case 2:
      case 3:
        simde_mm512_i32scatter_ps(HEDLEY_STATIC_CAST(void*, f32_buffer), vindex, a, 4);
        simde_test_x86_assert_equal_i32x16(simde_mm512_i32gather_ps(vindex, HEDLEY_STATIC_CAST(const void*, f32_buffer), 4), a);
        break;
      case 4:
      case 5:
      case 6:
      case 7:
        simde_mm512_i32scatter_ps(HEDLEY_STATIC_CAST(void*, f32_buffer), vindex, a, 8);
        simde_test_x86_assert_equal_i32x16(simde_mm512_i32gather_ps(vindex, HEDLEY_STATIC_CAST(const void*, f32_buffer), 8), a);
        break;
      default:
        return 1;
    }
  }
  return 0;
#else
  fputc('\n', stdout);

  (void)f32_buffer;

  for (int i = 0; i < 8; i++) {
    int32_t e[16];
    for (size_t i = 0; i < (sizeof(e) / sizeof(e[0])); i++) {
      size_t j;
      renew:
      e[i] = HEDLEY_STATIC_CAST(int32_t, (simde_test_codegen_random_u8()));
      for (j = 0; j < i; j++) if (e[j] == e[i]) goto renew;
    }
    simde__m512i vindex = simde_mm512_loadu_epi32(e);
    simde__m512 a = simde_test_x86_random_f32x16(SIMDE_FLOAT32_C(-1000.0), SIMDE_FLOAT32_C(1000.0));

    simde_test_x86_write_i32x16(2, vindex, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_f32x16(2, a, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

SIMDE_TEST_FUNC_LIST_BEGIN
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_i32scatter_epi32)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_mask_i32scatter_epi32)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_i32scatter_ps)
SIMDE_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
