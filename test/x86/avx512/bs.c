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
#include <simde/x86/avx512/bs.h>

static int
test_simde_mm512_bslli_epi128 (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[8];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C(  222463859744118245), -INT64_C( 8537180544267042865),  INT64_C( 8670161132952224993), -INT64_C( 5322000422564542841),
        -INT64_C( 5314119023781376630), -INT64_C( 8087867147652769869), -INT64_C( 1857438666390256092), -INT64_C( 4081913563613115948) },
      {  INT64_C(  222463859744118245), -INT64_C( 8537180544267042865),  INT64_C( 8670161132952224993), -INT64_C( 5322000422564542841),
        -INT64_C( 5314119023781376630), -INT64_C( 8087867147652769869), -INT64_C( 1857438666390256092), -INT64_C( 4081913563613115948) } },
    { { -INT64_C( 2356432447240619858),  INT64_C( 2979652346882614812), -INT64_C(  655796967272999776), -INT64_C( 5480048398313004051),
         INT64_C( 6320965061783910061),  INT64_C( 5379967202095124602), -INT64_C( 6510180067022564209), -INT64_C( 1959979295412633068) },
      {  INT64_C( 5495847938816519680),  INT64_C( 6474493779857775839), -INT64_C( 1863326958501978112), -INT64_C(  939840366203113994),
        -INT64_C( 5146422669759566592), -INT64_C( 6234201791864473001), -INT64_C( 6399130523916792064), -INT64_C( 3692609635476171611) } },
    { {  INT64_C( 2543255787379954872),  INT64_C( 8590983422739180613), -INT64_C( 3945486745152748319), -INT64_C( 8706110299316286190),
        -INT64_C( 6588830716187186493),  INT64_C( 7030343868882301746), -INT64_C( 7219489516489623216),  INT64_C( 2565663517082693957) },
      {  INT64_C( 8478575766923640832),  INT64_C( 5613718945715790667), -INT64_C( 3407649143728832512), -INT64_C( 5850376155700213442),
        -INT64_C( 4224538650269777920), -INT64_C( 3710937972943444849),  INT64_C( 4073793912342315008),  INT64_C( 1252023668868225999) } },
    { {  INT64_C( 1008277294043567867), -INT64_C( 5894447348705685167), -INT64_C( 6155151420511015824), -INT64_C( 8883532193012484044),
        -INT64_C( 5306968362551720336), -INT64_C( 6306805054355046291), -INT64_C( 2020648731289768730),  INT64_C( 9060167184679416750) },
      { -INT64_C( 2640640902636240896),  INT64_C( 6760228020743437854),  INT64_C( 7276043676432728064), -INT64_C( 8532985970471299966),
         INT64_C( 7227114421154217984),  INT64_C( 4020456078024530396), -INT64_C( 5815377279609470976),  INT64_C(  952642934805886262) } },
    { { -INT64_C( 1905547694197597324), -INT64_C( 5295905709828860254),  INT64_C(  569694740065362773),  INT64_C(   15195855065669290),
        -INT64_C( 4399059023625060713),  INT64_C( 2451116860508971901), -INT64_C( 6440130989075856064),  INT64_C( 8356872061336805484) },
      { -INT64_C( 2825556368679763968), -INT64_C( 2760681682686303625),  INT64_C(  288550699107614720), -INT64_C( 5242367356865874343),
         INT64_C( 1070166310970720256),  INT64_C( 5592243122847573792), -INT64_C( 7411230364355526656), -INT64_C(  477979228174807300) } },
    { { -INT64_C( 5227987322493873520),  INT64_C( 7712142971025821208), -INT64_C( 6291348105958511346),  INT64_C( 3590383341551029892),
        -INT64_C( 2135175943950346899),  INT64_C( 8933116826519823202),  INT64_C( 2728959960964005116), -INT64_C( 8640902333707485116) },
      { -INT64_C( 9092046368033210368),  INT64_C(  364255796041956016),  INT64_C(  300912143266480128), -INT64_C( 6841384923876279542),
        -INT64_C( 2209740195387932672), -INT64_C( 4079308113082791257), -INT64_C( 8791875295603851264), -INT64_C( 8826980340196794313) } },
    { { -INT64_C( 5955292201965586514), -INT64_C( 1380297277851426063), -INT64_C( 2813901404527894994), -INT64_C( 9212162710698165884),
        -INT64_C( 3116118331900799287),  INT64_C( 6580051720652651380), -INT64_C(  156467677292300952),  INT64_C( 3236051163989971350) },
      { -INT64_C(  599541700393697280),  INT64_C( 2517984271106013231),  INT64_C( 6786361688493916160),  INT64_C( 8756362113763823140),
         INT64_C( 6253529557580644352), -INT64_C(  327402946106813192),  INT64_C( 8460011900015476736), -INT64_C( 7955892604233974743) } },
    { { -INT64_C( 6354296280750956519),  INT64_C( 8540053104505914364),  INT64_C( 6938155635132509651), -INT64_C( 4510442265184246312),
        -INT64_C( 6599305448468404336), -INT64_C( 6479182519541598575), -INT64_C( 6473577230030960642),  INT64_C( 3713579231735837216) },
      {  INT64_C( 1801439850948198400), -INT64_C(  240994251960467232), -INT64_C( 3242591731706757120), -INT64_C( 2855201591067381075),
        -INT64_C( 8070450532247928832), -INT64_C( 7952113881080152665), -INT64_C(  144115188075855872),  INT64_C( 2352613192196813447) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi64(test_vec[i].a);
    simde__m512i r;

    SIMDE_CONSTIFY_8_(simde_mm512_bslli_epi128, r, (HEDLEY_UNREACHABLE(), a), i & 0xf, a);

    simde_test_x86_assert_equal_i64x8(r, simde_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    simde__m512i a = simde_test_x86_random_i64x8();
    simde__m512i r;

    SIMDE_CONSTIFY_8_(simde_mm512_bslli_epi128, r, (HEDLEY_UNREACHABLE(), a), i & 0xf, a);

    simde_test_x86_write_i64x8(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i64x8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

static int
test_simde_mm512_bsrli_epi128 (SIMDE_MUNIT_TEST_ARGS) {
#if 1
  static const struct {
    const int64_t a[8];
    const int64_t r[8];
  } test_vec[] = {
    { {  INT64_C( 3382090712519336405), -INT64_C( 7817876169626994992), -INT64_C( 2730082640474511180), -INT64_C( 7954808585498064915),
        -INT64_C( 2270660766135781704), -INT64_C( 1741824110368137164), -INT64_C( 7323808824001896139),  INT64_C( 1580509878743434658) },
      {  INT64_C( 3382090712519336405), -INT64_C( 7817876169626994992), -INT64_C( 2730082640474511180), -INT64_C( 7954808585498064915),
        -INT64_C( 2270660766135781704), -INT64_C( 1741824110368137164), -INT64_C( 7323808824001896139),  INT64_C( 1580509878743434658) } },
    { {  INT64_C( 3567096081481399952), -INT64_C( 3613500327318139969), -INT64_C( 7679292725886232539), -INT64_C( 7592257504013980367),
        -INT64_C( 6769272100106140191), -INT64_C( 1592181059496870014),  INT64_C( 1319319265320122055),  INT64_C(  482802312237040150) },
      { -INT64_C( 4669809643397029122),  INT64_C(   57942358384341451),  INT64_C( 3572882464685903704),  INT64_C(   42400338162873325),
        -INT64_C( 9033641723882031611),  INT64_C(   65838136774268287),  INT64_C( 1590420659714571318),  INT64_C(    1885946532175938) } },
    { {  INT64_C( 2507927337260280391), -INT64_C( 7413448555156970895),  INT64_C( 1895350252468427353),  INT64_C( 8579614782191562213),
         INT64_C( 6311637537901442260),  INT64_C( 5854023071245849810),  INT64_C( 1847277612532797572), -INT64_C( 7462969516236744249) },
      {  INT64_C( 7093489156019465187),  INT64_C(     168354728981820), -INT64_C( 1304607596304188589),  INT64_C(     130914532198967),
        -INT64_C( 3111328084613761630),  INT64_C(      89325303211148), -INT64_C( 2754204459894339444),  INT64_C(     167599099082531) } },
    { {  INT64_C( 6009390151989268776),  INT64_C( 2432283120271618680), -INT64_C( 1364226581363864123), -INT64_C( 7962553901840583388),
        -INT64_C( 6253243470251586460),  INT64_C( 3005266976862449910), -INT64_C( 3730331068387926745), -INT64_C( 2926720235304968088) },
      { -INT64_C(  580269102394474226),  INT64_C(        144975371376), -INT64_C( 2002653858680330504),  INT64_C(        624906430951),
         INT64_C( 1815221856480014821),  INT64_C(        179127870611), -INT64_C( 5539312315289851826),  INT64_C(        925065507793) } },
    { { -INT64_C(  244614919092436545), -INT64_C( 5354142727517785308),  INT64_C( 8076149336582640800), -INT64_C( 4996280910119516065),
         INT64_C( 9126620896317629862),  INT64_C( 2034705071845357816), -INT64_C( 2520193356871221362), -INT64_C( 7784937129914709951) },
      { -INT64_C( 8534405797405592570),  INT64_C(          3048358798), -INT64_C( 3295851665053958392),  INT64_C(          3131679995),
        -INT64_C( 8788993507862231719),  INT64_C(           473741691),  INT64_C( 4372793210918304126),  INT64_C(          2482395373) } },
    { { -INT64_C( 2335705856303193667), -INT64_C( 2037982917310440417), -INT64_C( 3081165195306263583),  INT64_C(  164390481882639560),
        -INT64_C( 8183130203726706775),  INT64_C( 6306130703346685002),  INT64_C( 5000952156865887740),  INT64_C( 2965413093023905760) },
      {  INT64_C( 6168635835838076391),  INT64_C(            14923681),  INT64_C( 5022940668081028480),  INT64_C(              149512),
         INT64_C( 8547878084509855662),  INT64_C(             5735392),  INT64_C( 9162005509237466867),  INT64_C(             2697027) } },
    { {  INT64_C( 4235680457489465803), -INT64_C( 3546200073126087940),  INT64_C( 5989160180797898842),  INT64_C( 2802822658972457020),
        -INT64_C( 4464710146708168991), -INT64_C( 7152562544770161637),  INT64_C( 2037194801781414782),  INT64_C( 4551940059893907113) },
      {  INT64_C( 6760592275341589192),  INT64_C(               52937), -INT64_C( 6891707580771708131),  INT64_C(                9957),
        -INT64_C(  125277023896878582),  INT64_C(               40124), -INT64_C( 4801394823772169147),  INT64_C(               16171) } },
    { { -INT64_C( 3466616452217336997), -INT64_C( 7575765857389962754), -INT64_C( 8389979733289840236), -INT64_C( 9067793173420480117),
         INT64_C( 5821614258791856198), -INT64_C( 7760144158537185376),  INT64_C( 3399259647306684095), -INT64_C( 4851993672803632383) },
      { -INT64_C( 2487931752327545137),  INT64_C(                 150),  INT64_C( 2934700891760593803),  INT64_C(                 130),
         INT64_C( 5651455375112118352),  INT64_C(                 148), -INT64_C( 6178527299189931729),  INT64_C(                 188) } },
  };

  for (size_t i = 0 ; i < (sizeof(test_vec) / sizeof(test_vec[0])) ; i++) {
    simde__m512i a = simde_mm512_loadu_epi64(test_vec[i].a);
    simde__m512i r;

    SIMDE_CONSTIFY_8_(simde_mm512_bsrli_epi128, r, (HEDLEY_UNREACHABLE(), a), i & 0xf, a);

    simde_test_x86_assert_equal_i64x8(r, simde_mm512_loadu_epi64(test_vec[i].r));
  }

  return 0;
#else
  fputc('\n', stdout);
  for (int i = 0 ; i < 8 ; i++) {
    simde__m512i a = simde_test_x86_random_i64x8();
    simde__m512i r;

    SIMDE_CONSTIFY_8_(simde_mm512_bsrli_epi128, r, (HEDLEY_UNREACHABLE(), a), i & 0xf, a);

    simde_test_x86_write_i64x8(2, a, SIMDE_TEST_VEC_POS_FIRST);
    simde_test_x86_write_i64x8(2, r, SIMDE_TEST_VEC_POS_LAST);
  }
  return 1;
#endif
}

SIMDE_TEST_FUNC_LIST_BEGIN
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_bslli_epi128)
  SIMDE_TEST_FUNC_LIST_ENTRY(mm512_bsrli_epi128)
SIMDE_TEST_FUNC_LIST_END

#include <test/x86/avx512/test-avx512-footer.h>
