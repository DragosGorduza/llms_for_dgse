# LLMs in Macroeconomic Modelling: A Literature Review

Oct 14, 2026 · @Dragos

## Scope and taxonomy

The literature splits cleanly by *where the LLM sits in the modelling chain*: as the behavioural core of a model (agents), as the whole model (forecasts), as a measurement device feeding a conventional model (plug-in variables), or as a generator of the conditioning paths a model is run on (scenarios). The evidence base is strongest for the first and third, most contested for the second, and thinnest for the fourth.

| Category | What the LLM replaces | Typical output | Model it feeds | Main validity threat |
| --- | --- | --- | --- | --- |
| 1. Stand-in for agents | Behavioural rules, rational-expectations operator, survey respondents | Decisions, beliefs, votes | Agent-based models, survey experiments, committee models | Homogeneity; persona sensitivity; behaviour that is more "textbook" than human |
| 2. One-stop forecasts | The forecasting model itself | Point/density forecasts of CPI, GDP, unemployment, rates | None (end product) | Look-ahead bias / memorisation; stale real-time forecasts |
| 3. Plug-in values | Survey, hand-coding or a data series | Indices, shocks, expectations series, parameters | VARs, DSGE, Phillips curves, nowcasting models | Measurement error and leakage that propagate into estimates |
| 4. Scenario generation | Staff judgement in writing narratives and paths | Narrative scenarios, conditioning paths, counterfactual policy rules | Stress tests, conditional forecasts, policy simulators | No ground truth; plausibility hard to score |

**Coverage.** Working papers and peer-reviewed articles from 2023 to October 2026, with priority on central-bank, NBER, arXiv econ/q-fin and ACL/NeurIPS venues. Pre-LLM text-as-data work (dictionary, topic-model, BERT) is cited only where it is the direct benchmark for an LLM study. Pure financial-market return prediction is excluded unless the paper uses macro variables or macro agents.

**Boundary cases.** Several papers sit across categories. Synthetic survey panels (e.g. simulated SPF) are classed under agents when the persona is the object of study and under forecasts when accuracy against outturns is the headline. LLM-generated expectations series are classed under plug-in values when they are used as regressors in an existing model.

## 1. LLMs as stand-ins for agents

LLM agents reproduce *qualitative* macro regularities and the *direction* of survey treatment effects well, but they are systematically more homogeneous, more "textbook-rational" and more prompt-dependent than humans. The most credible designs condition agents on real priors (lagged survey medians, persona histories) and validate against a human benchmark before running counterfactuals.

The motivation is the expectations problem in modern macro. [Moll (2025)](https://www.tse-fr.eu/seminars/2025-trouble-rational-expectations-heterogeneous-agent-models-challenge-macroeconomics) argues rational expectations in heterogeneous-agent models are computationally implausible and calls for replacements that are tractable, empirically grounded and partly Lucas-critique-proof; LLM agents are one candidate, alongside learning and survey-based expectations. The conceptual anchor is [Horton (2023; revised 2026 with Filippas and Manning)](https://arxiv.org/abs/2301.07543v2), which frames LLMs as *Homo silicus*: implicit computational models of humans that can be endowed with preferences and information and simulated.

### 1a. Expectation formation: households, firms, professionals

| Paper | Agents simulated | Benchmark | Headline result |
| --- | --- | --- | --- |
| [Wu, Xi & Xie (2025), *LLM Survey Framework*](https://www.econ.cuhk.edu.hk/wp-content/uploads/2025/11/XieShihan.pdf) | 200 SCE household personas, GPT-4.1 / GPT-5 | Weber et al. (2025) multi-wave inflation RCT | Recovers state-dependent treatment effects at \~1/1000 of survey cost; priors track SCE medians (slope 1.05) and correlate 0.84 with Michigan; extends 10 waves to 50+ back to 1990; treatment effects halve in \~3 months |
| [Anesti, Hill & Joseph (BoE SWP 1190, 2026)](https://www.bankofengland.co.uk/-/media/boe/files/working-paper/2026/inflation-attitudes-of-large-language-models.pdf) | UK households (Inflation Attitudes Survey personas), GPT-3.5 | IAS around the 2022 peak; uses Sept-2021 training cutoff as a natural experiment | Tracks short-horizon survey aggregates; reproduces income, tenure and class gradients and food-price salience, but lacks a consistent model of CPI |
| [Lin et al. (2025)](https://arxiv.org/abs/2505.17648v2) | Thousands of household and expert agents with persona, prior and knowledge modules | Survey experiments on inflation and unemployment | Captures key heterogeneity and drivers, but distributions are more homogeneous than humans; the prior-expectations module matters most |
| [Zarifhonarvar (2024)](https://marginalrevolution.com/marginalrevolution/2024/05/experimental-evidence-on-large-language-models.html) | Inflation expectations under six knowledge sources | Human information-treatment results | Domain-relevant documents reduce response variance; mortgage-rate treatments move priors as in human surveys |
| [Tariff-threats multi-agent study (2026)](https://huggingface.co/papers/2608.30522) | 300 Michigan-survey households as persistent agents exposed to social media | Post-"Liberation Day" survey data | Reproduces some distributional and demographic patterns; message features and central-bank explanations shape beliefs and dispersion |
| [BIS ACC network paper on Indonesia (2026)](https://www.bis.org/events/accconf2026/Paper_1_ID_251216.pdf) | 5,400 household personas | Fills a discontinued national survey | LLM panel used as a substitute instrument for information-treatment experiments |
| [Hansen, Horton, Kazinnik, Puzzello & Zarifhonarvar, *Simulating the SPF*](https://john-joseph-horton.com/papers/simulating-the-survey-of-professional-forecasters/) | SPF panellists from hand-collected bios; 23 variables, 1999–2023 | Philadelphia Fed SPF | Replicates accuracy, revisions and dispersion; ablations show lagged SPF medians and real-time data drive performance, personas only modestly |
| [*CFOs Meet LLMs* (2026)](https://arxiv.org/pdf/2606.13812) | Individual CFOs from the Duke/Richmond–Atlanta CFO Survey (6,075 matched responses) | Same CFO's actual answer | LLM "digital twins" significantly predict firm-quarter optimism |
| [Bybee (2023)](https://arxiv.org/pdf/2305.02823) | A generic reader of WSJ news, 1984–2021 | SPF, AAII, Duke CFO surveys | Generated expectations match surveys *and* their deviations from FIRE (under-reaction, return extrapolation); correlation persists outside the training window |

### 1b. LLM agents inside macro simulations

- [Li et al., *EconAgent* (ACL 2024)](https://aclanthology.org/2024.acl-long.829) is the reference build: household agents with perception and memory modules choose work and consumption; the economy reproduces a Phillips curve and Okun's law better than rule-based or RL agents.
- [*Shachi* (2025)](https://arxiv.org/pdf/2509.21862) re-runs EconAgent (100 agents, 240 months) across backend models. All produce Phillips and Okun relationships, but intercepts and slopes shift with the model, so the LLM itself is a structural parameter.
- [Feng et al., *SimCity* (2025)](https://arxiv.org/abs/2510.01297v1) adds LLM firms, a central bank and a government with labour, goods and financial markets, and passes a checklist of five stylised facts (Engel, Okun, Phillips, Beveridge, price elasticity).
- [Hao & Xie, *MLAB* (2025)](https://arxiv.org/html/2502.16879v1) maps *different* LLMs to education and income groups to build heterogeneity for tax-policy simulation.
- [del Rio-Chanona, Pangallo & Hommes (2025)](https://arxiv.org/abs/2505.07457v1) port learning-to-forecast experiments to interacting LLM agents: they show bounded rationality and reproduce the positive- vs negative-feedback distinction, but with less heterogeneity than humans.
- [Gusella & Vicario (2025)](https://ideas.repec.org/p/arx/papers/2511.08604.html) let an LLM choose between fundamentalist and trend-following rules on S&P 500 data (1990–2020); switching shares match the heterogeneous-agent-model literature, with a tilt toward fundamentalism.

### 1c. Policymakers and financial-stability agents

- [Kazinnik & Sinclair, *FOMC in Silico*](https://www.aeaweb.org/conference/2027/preliminary/1349) simulates each FOMC member from biographies and real-time data, then lets them deliberate and vote. In a conditional backtest over 215 meetings (2000–2026) it is within 25 bp in 93% of meetings (MAE 9.1 bp) and tracks the funds rate better than Taylor-rule benchmarks; political-pressure scenarios raise dissent.
- [Kazinnik, *Bank Run, Interrupted* (Management Science)](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2024.06807) estimates depositor withdrawal propensities from persona surveys, validates against human benchmarks, and propagates them through a contagion network to test crisis communications.
- [Hansen & Lee (FEDS 2025-090)](https://www.federalreserve.gov/econres/feds/financial-stability-implications-of-generative-ai-taming-the-animal-spirits.htm) replicate herding experiments: LLM agents herd less than humans, but can be steered into herding and retain some human biases.
- [Henning et al. (2025), *LLM agents do not replicate human market traders*](https://arxiv.org/pdf/2502.15800v3) find LLMs trade close to fundamental value where humans generate bubbles, a warning against using them as stand-ins for market psychology.

### What the evidence says about validity

1. **Direction right, dispersion wrong.** Nearly every validation finds correct signs and rankings but compressed cross-sectional variance (Lin et al.; del Rio-Chanona et al.).
2. **Conditioning beats persona.** Performance comes mostly from feeding real priors and data, not from demographic role-play (Hansen et al.; Lin et al.).
3. **Rationality bias cuts both ways.** LLMs often look more rational than people (Hansen & Lee; Henning et al.), which helps forecasting but undermines behavioural realism.
4. **The backend is a parameter.** Swapping models moves simulated aggregates (Shachi), so results need multi-model robustness.

## 2. LLMs as one-stop-shops for forecasts

The headline pattern is a **gap between retrospective and real-time evidence**. Prompted LLMs look as good as or better than the SPF in pseudo out-of-sample tests inside their training window, but genuine real-time forecasts are stale and anchored, and time-series foundation models roughly match, rather than beat, BVARs and factor models. The direct-forecast use case is therefore the least mature of the four.

### 2a. Prompted general-purpose LLMs (text in, number out)

| Paper | Target / country | Design | Result |
| --- | --- | --- | --- |
| [Faria-e-Castro & Leibovici (St. Louis Fed Review, 2024)](https://www.stlouisfed.org/publications/review/2024/nov/artificial-intelligence-and-inflation-forecasts) | US CPI, 2019–2023 | PaLM, in-sample conditional density forecasts vs SPF | Lower MSE than SPF in most years and horizons; slower reversion to the 2% anchor |
| [Pham & Cunningham (2024)](https://arxiv.org/abs/2404.07396v2) | US inflation and unemployment, 2022 | GPT-3.5/4, direct vs "future narrative" prompts (e.g. a Powell speech set in the future) | Narrative prompts improve GPT-4 accuracy; injecting the Ukraine invasion worsens it |
| [Iadisernia & Camassa (2025)](https://ideas.repec.org/p/arx/papers/2511.02458.html) | Euro-area HICP, core, GDP, unemployment; 50 ECB-SPF rounds 2013–2025 | GPT-4o with 2,368 personas plus out-of-sample 2024–25 test | Competitive out-of-sample when given context data; personas add nothing; forecasts far more homogeneous than the human panel |
| [Alam, Boyle, Li & Sekhposyan, *ChatMacro* (FRBSF WP 2026-04)](https://www.aeaweb.org/conference/2027/preliminary/1349) | US inflation | Real-time elicitation vs pseudo out-of-sample | Pseudo-OOS forecasts match benchmarks; genuine real-time forecasts are largely inaccurate and stale |
| André, Bessec & Goulby (2025), [Banque de France practitioner study](https://isor.univie.ac.at/isor-colloquium/talk/news/llms-vs-econometric-models-for-nowcasting-gdp-growth-a-practitioners-view) | French GDP nowcasts | Zero-shot LLMs vs BdF econometric suite | Econometric models win in calm periods; LLMs do better in exceptional episodes such as Covid; prompt language and model version matter |
| [Bernasconi, Gómez García & Manzano Quiroga (2025)](https://aaep.org.ar/works/works2025/4806.pdf) | Argentine GDP, 2021–2024 | Gemini with live web access vs central-bank analyst survey (REM), real time | Competitive nowcasts when combined with search |
| Gonzalez (2026); Giannellis et al. (*J. Forecasting*, 2026); Benny et al. (2025) — as reviewed by [Garavito-Acosta et al.](https://repec.graduateinstitute.ch/pdfs/Working_papers/HEIDWP23-2026.pdf) | Argentina, US/euro area, India | Retrospective one-step and annual forecasts | Beat random walk and simple AR in calm regimes; miss the size of spikes and turning points |
| [Garavito-Acosta, Martínez-Rivera, Ospina-Tejeiro & Caicedo-García (Banco de la República, 2026)](https://repec.graduateinstitute.ch/pdfs/Working_papers/HEIDWP23-2026.pdf) | Colombian CPI, 24 months | GPT-4o/4.1 and Gemini 2.5 Pro run daily in real time across six information-set steps | Bare prompts revert mechanically to target; adding theory rules and web-retrieved shocks yields persistent, survey-like paths; accuracy evaluation pending |

The [SPF simulation of Hansen et al.](https://john-joseph-horton.com/papers/simulating-the-survey-of-professional-forecasters/) (Section 1a) belongs here too: its accuracy comes mostly from feeding lagged SPF medians, i.e. the LLM is a smart combiner of human forecasts rather than an independent model.

### 2b. Time-series foundation models (numbers in, numbers out)

- [Carriero, Pettenuzzo & Shekhar (2024–25)](https://arxiv.org/abs/2407.00890v3) benchmark time-series language models on FRED-MD: the best are broadly comparable to, or slightly worse than, Bayesian VARs and factor models, with less stable gains; fine-tuning does not help much. The August 2026 revision of the paper adds that TSFM and econometric forecast errors are nearly orthogonal at short horizons, so the two are complements, and flags that three of five TSFMs list many of the target series in their training data.
- [Koyuncu, Kwon, Lombardi, Pérez-Cruz & Shin, *BISTRO* (BIS WP 1337, 2026)](https://www.bis.org/publ/work1337.htm) fine-tune a foundation model on BIS macro data; it gives reliable unconditional forecasts and supports *conditional* forecasts that reveal nonlinearities (see Section 4).
- [Dass, Racheter, Scheufele, Stucki, Vollmer & Winter (SNB, 2026)](https://www.bankofengland.co.uk/-/media/boe/files/events/2026/y-stucki-slides.pdf) run a true real-time test since June 2025 on 11 daily FX, yield and equity series: all TSFM families have median relative RMSFE above 1 against ARIMA, only TimeGPT stays close, and fine-tuning beats zero-shot in just 27% of head-to-heads.
- A [Massey University study of New Zealand GDP](https://mro.massey.ac.nz/handle/10179/73995) finds Chronos, TimeGPT and Moirai match or beat RBNZ-style baselines zero-shot, degrade in shocks but recover faster.
- [Chib & Tan (2025)](https://repec.graduateinstitute.ch/pdfs/Working_papers/HEIDWP23-2026.pdf) train a macro transformer on observed plus DSGE-simulated data, a bridge between structural models and foundation models.

### 2c. Retrieval-augmented and agentic forecasting

General forecasting research supplies the method frontier: Halawi et al. (NeurIPS 2024) approach crowd-level accuracy with retrieval plus aggregation, while ForecastBench (Karger et al., ICLR 2025) shows superforecasters still beat the best LLMs on unresolved questions. In macro, the Colombian and Argentine studies above are the first to embed live search into a central-bank forecasting pipeline.

### Assessment

1. **In-sample wins do not survive.** Every study that moves from pseudo to real time (ChatMacro, SNB) reports a sharp deterioration.
2. **LLMs are good aggregators, weak extrapolators.** They add value when fed human forecasts and context, and default to mean reversion toward targets when not.
3. **Comparative advantage at breaks.** Several papers find LLMs relatively better in unusual episodes (Covid), where judgement and narrative matter, and worse in routine periods.
4. **Homogeneity.** Persona diversity does not create forecast diversity, limiting their value as synthetic panels for disagreement measures.

## 3. LLMs to plug in values for variables in existing models

This is the most productive and least controversial category: the LLM acts as a **measurement instrument** that turns text (news, minutes, earnings calls, PMI releases) into a series that slots into a VAR, local projection, nowcasting model or Phillips curve, while the economics stays in the conventional model. The main risks are generated-regressor error and information leakage into the series itself.

### By model slot

| Slot in the model | Paper | What the LLM produces | Downstream use and result |
| --- | --- | --- | --- |
| Monetary policy shock (VAR/LP instrument) | [Hansen & Kazinnik, *Can ChatGPT Decipher Fedspeak?*](https://www.frbsf.org/research-and-insights/publications/system-research-richmond-fed/2024/03/can-chatgpt-decipher-fedspeak/) | Hawkish/dovish stance of FOMC statements; Romer–Romer-style narrative shocks with GPT-4 | Beats BERT and dictionaries; explanations comparable to human analysts |
| Monetary policy shock | [Fernández-Fuertes, *Monetary Policy Shocks: A New Hope* (Bocconi WP 25257, 2025)](https://econpapers.repec.org/paper/bafcbafwp/cbafwp25257.htm) | Multi-agent LLM reads Beige Books and Minutes before each meeting to form expectations; surprise = decision minus expectation | Less noisy than high-frequency surprises; contractionary shocks give persistent disinflation |
| Monetary policy shock (euro area) | [Betz, Dix & Streit (CEPR DP 21390)](https://cepr.org/publications/DP21390) | Classifies each ECB decision from 7,620 articles in 11 newspapers as expected, unexpectedly tight or loose | Aligns with HFI series except in financial turmoil; mostly free of information effects |
| Supply vs demand decomposition | [Kermani et al., *In the Fed's Mind* (2025)](https://www.lse.ac.uk/economics/Assets/Documents/AMIR-Kermani-In-the-Feds-Mind.pdf) | Tags reasons cited in FOMC records (1936–2025) as supply- or demand-driven | Shows the Fed reacts more aggressively to demand-driven inflation |
| Activity and inflation sentiment (nowcasting regressor) | [Kwon, Park, Rungcharoenkitkul & Smets (BIS WP 1294, 2025)](https://www.bis.org/publ/work1294.htm) | Growth and inflation sentiment from US news, decomposed into demand, supply and structural drivers | Tracks hard data and improves out-of-sample forecasts of simple models |
| GDP nowcast regressor | [de Bondt & Sun (ECB WP 3063, 2025)](https://www.suerf.org/publications/suerf-policy-notes-and-briefs/gdp-nowcasting-power-of-pmi-news-releases-using-chatgpt/) | ChatGPT text scores from PMI press releases | Improves euro-area GDP nowcasts against Eurosystem projections and the flash estimate; gains vary by year |
| Inflation nowcast regressor | [Allard, Teiletche & Zinebi (2024)](https://arxiv.org/html/2410.20198v1) | Fine-tuned InflaBERT news-sentiment index | Added to the Cleveland Fed nowcasting model |
| Expectations series | [Jha, Qian, Weber & Yang (2024–25)](https://www.arxiv.org/abs/2410.03897) | AI Economy Score from 120,000+ earnings-call transcripts | Predicts GDP, production and employment up to 10 quarters ahead, beyond survey forecasts; used in a VAR |
| Expectations series | [Bybee (2023)](https://arxiv.org/pdf/2305.02823); [Wu, Xi & Xie (NBER WP 34308)](https://www.econ.cuhk.edu.hk/wp-content/uploads/2025/11/XieShihan.pdf) | Back-filled belief series from news, or retrospective household panels to 1990 | Extends short survey samples so expectation-augmented models can be estimated over longer periods |
| Behavioural parameters (ABM calibration) | [Yee & Sharma (2026)](https://arxiv.org/abs/2602.01022v1) | Loss aversion, herding, extrapolation, disposition from 24,000 agent-scenario pairs | Baseline LLMs are too rational (loss aversion 1.12–1.90 vs a human 2.25); profile prompts recalibrate them; calibrated extrapolation reproduces momentum and reversal in an ABM |

Also relevant: Jaworski (*J. Forecasting*, 2026) feeds a GPT-4o social-media inflation-expectations indicator into ARIMAX models, and Tawichsri et al. (PIER DP 252, 2026) build LLM-assisted news-based expectations, both reviewed in [Garavito-Acosta et al.](https://repec.graduateinstitute.ch/pdfs/Working_papers/HEIDWP23-2026.pdf). The pre-LLM benchmark for language-based shock identification is [Aruoba & Drechsel](https://nber.org/system/files/working_papers/w32417/w32417.pdf).

### 3b. Inside structural NK, DSGE and HANK models

As of October 2026 I found **no paper that estimates or simulates a Galí-style three-equation New Keynesian model, or a HANK model, with LLM-generated expectations or parameters substituted into its equations**. The closest work fills one slot of the model at a time, or runs the link in the opposite direction.

| Slot in the 3-equation NK model | Closest work | What it does | Status |
| --- | --- | --- | --- |
| Expectations in the Phillips curve | [Wu, Xi & Xie](https://www.econ.cuhk.edu.hk/wp-content/uploads/2025/11/XieShihan.pdf); [Hansen et al.](https://john-joseph-horton.com/papers/simulating-the-survey-of-professional-forecasters/); [Bybee](https://arxiv.org/pdf/2305.02823) | Build LLM inflation-expectation series (household panels back to 1990; SPF-style panels; news-based beliefs) | Gap: no NKPC yet estimated with them, though survey expectations are known to fix NKPC puzzles ([Coibion, Gorodnichenko & Kamdar, JEL 2018](https://www.nber.org/papers/w23304.pdf)) |
| Expectations in the dynamic IS curve | [Lin, Sun & Yan (2025)](https://arxiv.org/pdf/2505.17648v2) | Re-run Andre et al. (2022) "subjective models" vignettes (oil, rate, spending, tax shocks): \~93% of expert agents say a rate hike lowers inflation, while \~33% of household agents say it raises it | Tests agents' model of the economy against the textbook NK answer; not embedded in an IS curve |
| Behavioural expectation parameters | [Wang & Liu (2026), Behavioral Kalman Filter](https://arxiv.org/pdf/2601.17527) | Estimate prior weights (0.48–0.89) and signal gains for LLM household and CEO agents; weights do not sum to one, and concurrent signals interfere; LoRA fine-tuning reduces but does not remove this | Natural input for Gabaix-style cognitive discounting or sticky-information NK models, but not plugged into one |
| Behavioural parameters (other) | [Yee & Sharma (2026)](https://arxiv.org/abs/2602.01022v1) | Calibrate loss aversion, herding, extrapolation from LLM agents | Embedded in an agent-based asset-pricing model, not NK |
| Taylor rule / policy block | [Kazinnik & Sinclair](https://www.aeaweb.org/conference/2027/preliminary/1349) | Replace the rule with a deliberating LLM committee; tracks the funds rate better than Taylor-rule benchmarks | Policy block only; rest of the economy is data, not a model |
| Taylor rule / policy block | [Dubey, Gupta & Agarwal (Oct 2026)](https://arxiv.org/html/2610.01128) | A fine-tuned Qwen2.5-1.5B sets policy inside six Snowdrop DSGE models, including Galí–Smets–Wouters, Ireland's (2004) NK model and Smets–Wouters, rewarded by the model's loss | Closest to a true NK embedding, but the LLM is the controller, not the expectations; confirmatory results still marked TBD |
| Whole model (reverse direction) | [Chib, Tan & Zhang (2026)](https://arxiv.org/pdf/2512.21031) | Sample a DSGE posterior to generate millions of synthetic panels that train a transformer forecaster | Matches or beats a VAR; DSGE disciplines the LLM rather than vice versa |
| HANK | — | Moll's critique lists survey expectations, learning and RL as replacements for rational expectations, not LLMs; LLM-agent economies (EconAgent, SimCity) are heterogeneous but not solved as HANK general equilibria | Open |

One caution: [Chatterjee & Kundu (2025)](https://ir.vidyasagar.ac.in/bitstream/123456789/7958/1/A4.pdf) present an NK-DSGE "with AI-driven expectations", but their AI agents use a recursive least-squares learning rule, not an LLM. The most direct feasible next step is to estimate an expectations-augmented NKPC, or calibrate a cognitive-discounting parameter, using LLM survey panels validated against the human surveys they replicate.

### Assessment

1. **Highest return per unit of risk.** The structural model is unchanged and auditable; the LLM only replaces hand-coding or a missing survey.
2. **Generated-regressor problem.** Ludwig, Mullainathan & Rambachan (NBER WP 33344) show that using LLM outputs in downstream estimation needs a validation sample of human labels to correct for measurement error; few macro papers do this.
3. **Leakage into the series.** [Dunn, Meade, Sinha & Kabir](https://www.aeaweb.org/conference/2027/preliminary/1349) find LLM topic classification of FOMC minutes degrades sharply on post-2025 text, so historical indices may embed hindsight.
4. **Parameters are prompt-dependent.** Yee & Sharma show calibration moves parameters by design, so an "LLM-measured" parameter is a function of the profile chosen, not a primitive.

## 4. LLMs for scenario generation

Scenario generation is the thinnest and least validated strand: most papers are proofs of concept, and none can be scored against outturns because scenarios are not forecasts. Two designs exist. In the first, the LLM writes the scenario itself (narrative plus shock sizes or paths). In the second, an LLM-agent simulator from Section 1 is re-run under alternative policy or communication conditions, so the scenario is the counterfactual input rather than the output.

### 4a. LLM-written scenarios and conditioning paths

| Paper | Domain | What is generated | How it is checked |
| --- | --- | --- | --- |
| [Soleimani (2025)](https://arxiv.org/abs/2512.07867) | G7 macro-financial stress testing | GPT-5-mini and Llama-3.1 write JSON shocks to GDP, inflation and policy rates for Q4 2026 off IMF WEO baselines, with narratives; mapped to portfolio VaR/ES | Two-layer plausibility audit and a regime classifier; deterministic "2008-style" benchmarks; risk variance is driven by portfolio and prompt design, not retrieval |
| [Garavito-Acosta et al. (Banco de la República, 2026)](https://repec.graduateinstitute.ch/pdfs/Working_papers/HEIDWP23-2026.pdf), Step 5 | Colombian inflation, FX, policy rate, GDP | LLM retrieves current shocks by web search, sizes them within literature ranges, and builds one joint macro scenario | Phillips-curve, Taylor-rule, UIP and shock-consistency rules; up to four automated revisions; best-scoring candidate kept |
| [Koyuncu et al., *BISTRO* (BIS, 2026)](https://www.bis.org/publ/work1337.htm) | Macro aggregates | Conditional forecasts given an assumed path for some variables | Used to reveal nonlinear responses; the BIS pitches it explicitly for scenario exploration |
| [Ross & Ross (2026)](https://arxiv.org/pdf/2603.29470) | Germany's energy transition | Simulated expert panel agrees 15 descriptors with 3 states each, cross-impacts, a preferred pathway and model-ready inputs | Cross-impact balance and multi-criteria analysis run inside the AI panel |
| [Badekale & Akinfaderin (2025)](https://arxiv.org/abs/2505.18694v1) | Sub-Saharan Africa climate policy | Policy scenarios drawn from COP documents | Embedding-based scores plus one human expert and two other LLMs as judges |
| [Pham & Cunningham (2024)](https://arxiv.org/abs/2404.07396v2) | US inflation and unemployment | "Future narratives" (scenes set after the cutoff) used as a forecasting device | Accuracy vs outturns; injecting a named event shifts the paths |

### 4b. Counterfactual runs of LLM-agent simulators

- **Policy-committee scenarios.** [FOMC in Silico](https://digitaleconomy.stanford.edu/publication/fomc-in-silico-a-multi-agent-system-for-monetary-policy-decision-modeling) reruns the July 2025 meeting under political pressure (dissent in 88% of runs, average 4.38%) and a negative jobs revision (4.30%, dissent 74%), against a Bayesian-voting Monte Carlo baseline.
- **Communication scenarios.** [Kazinnik's bank-run study](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2024.06807) generates message variants by tone and content and propagates withdrawals through a network; the [tariff-threats study](https://huggingface.co/papers/2608.30522) varies message immediacy, complexity and sender; [Collodel (Central Bank of Malta, 2025)](https://arxiv.org/html/2508.13635v2) simulates 30 synthetic traders across 283 ECB press conferences, reaching a correlation of about 0.5 with actual market disagreement at medium and long maturities.
- **Information scenarios.** [Wu, Xi & Xie](https://www.econ.cuhk.edu.hk/wp-content/uploads/2025/11/XieShihan.pdf) run "clean" treatments using data not yet released at the survey date, something impossible with human panels.
- **Fiscal scenarios.** EconAgent, SimCity and MLAB (Section 1b) run tax and transfer changes through LLM economies.

### Assessment

1. **No scoring rule.** Plausibility is judged by consistency rules or by other LLMs, which is partly circular.
2. **Best practice is "LLM proposes, model disposes".** The LLM supplies the narrative and shock sizes; a VAR, DSGE or stress-test satellite model turns them into paths. Soleimani and the Colombian pipeline approximate this; no published paper yet couples LLM narratives to a central bank's full structural model.
3. **Severity risk.** Bare LLMs mean-revert toward targets (Colombia Steps 0–3), so unconstrained adverse scenarios may be too mild.
4. **Lucas critique for agent counterfactuals.** Agents' responses to new policy regimes are only as credible as their out-of-sample validation, which Section 1 shows is mostly qualitative.

## 5. Cross-cutting methodological issues

Look-ahead bias is the single biggest threat and cuts across all four categories: any historical evaluation inside a model's training window is, by default, not identified. Prompt instructions to "ignore later information" demonstrably fail.

| Threat | Evidence | Mitigation used in the literature |
| --- | --- | --- |
| Memorisation / look-ahead bias | [Lopez-Lira, Tang & Zhu](https://papers.ssrn.com/abstract=5217505) prove forecasting skill is non-identified when realised values were seen; LLMs recall exact pre-cutoff macro values, and masking fails. [Eliseev & Seleznev (Bank of Russia WP 167, 2026)](https://www.cbr.ru/statichtml/file/188046/wp_167.pdf) propose "fake-date" tests: Kimi-K2, Qwen3 and DeepSeek-V3.1 all fail them | Evaluate only after the cutoff; real-time vintages (Colombia, SNB); anonymise entities and dates; chronologically consistent models trained year by year (He et al., 2025); fake-date tests as a diagnostic |
| Pseudo vs real out-of-sample gap | [Alam et al., *ChatMacro*](https://www.aeaweb.org/conference/2027/preliminary/1349): good pseudo-OOS, stale real-time forecasts | Report real-time vintages separately; never pool with retrospective tests |
| Vintage confusion | Crane, Karra & Soto (FEDS 2025-044, *Total Recall?*): LLMs mix first releases with revised data and misdate releases | Supply real-time data in the prompt rather than relying on recall |
| Leakage into text-derived series | [Dunn, Meade, Sinha & Kabir](https://www.aeaweb.org/conference/2027/preliminary/1349): FOMC-minutes topic classification collapses on post-2025 text | Hold-out on genuinely new text; human-labelled validation sets |
| Training-set contamination in TSFMs | [Carriero et al. (2026 revision)](https://arxiv.org/pdf/2407.00890): three of five TSFMs list target series in training data; [SNB](https://www.bankofengland.co.uk/-/media/boe/files/events/2026/y-stucki-slides.pdf) notes recursive re-estimation is infeasible | Real-time evaluation since release date |
| Prompt and persona sensitivity | Schoenegger et al. (2025): most prompt-engineering tricks add little and some hurt; André et al.: prompt language changes accuracy; [Iadisernia & Camassa](https://ideas.repec.org/p/arx/papers/2511.02458.html): personas add nothing | Pre-register prompts; report multiple prompts; ablate persona modules |
| Homogeneity and representativeness | Bisbee et al. (*Political Analysis*, 2024) and [Li et al. (2025)](https://arxiv.org/pdf/2503.16527) show synthetic samples understate variance and amplify bias; Section 1 papers find the same in macro | Condition on real priors; compare cross-sectional moments, not just means |
| Backend dependence and reproducibility | [Shachi](https://arxiv.org/pdf/2509.21862): Phillips-curve intercepts and Okun slopes shift with the LLM; zero-temperature output is still non-deterministic | Multi-model robustness; log model versions, seeds and dates; snapshot artefacts (as in Soleimani) |
| Generated regressors | Ludwig, Mullainathan & Rambachan (NBER WP 33344) formalise the no-leakage condition and the need for validation data when LLM outputs enter estimation | Small human-labelled validation sample to debias downstream coefficients |

The field's evaluation standard is converging on three rules: test only on post-cutoff or real-time data, benchmark against simple statistical and institutional forecasts (not just other LLMs), and publish prompts, model versions and dates.

## 6. Synthesis and open questions

The ranking by readiness is: plug-in variables first, agents second, scenarios third, one-stop forecasts last. The common thread is that LLMs add most where they replace a costly or missing *human* input (coding text, answering surveys, drafting narratives) inside a model whose economics is still explicit, and least where they replace the model itself.

| Category | Maturity | Strongest evidence | Main weakness | Where it is useful now |
| --- | --- | --- | --- | --- |
| 3. Plug-in values | High: central-bank working papers, published shock series | Narrative shocks, sentiment and expectations indices that improve out-of-sample nowcasts (BIS, ECB, Jha et al.) | Generated-regressor error; leakage in historical indices | Nowcasting regressors, shock instruments, back-filling short surveys |
| 1. Agent stand-ins | Medium: replicates human RCTs and survey moments | Wu, Xi & Xie; Hansen et al.; FOMC in Silico backtest | Too homogeneous and too rational; backend-dependent | Piloting survey and communication experiments; committee and run dynamics |
| 4. Scenario generation | Low: proofs of concept | Soleimani; Colombian Step-5 pipeline | No scoring rule; risk of mild, mean-reverting scenarios | Drafting narratives and shock menus for a structural model to quantify |
| 2. One-stop forecasts | Low in real time despite strong in-sample claims | Hansen et al. (as aggregator); Carriero et al. (TSFMs as complements) | Look-ahead bias; stale real-time forecasts; TSFMs do not beat BVARs | Combination input and judgemental cross-check in unusual episodes |

### Open research questions

1. **Calibrating heterogeneity.** How to make LLM agent populations match the *dispersion* of survey expectations, not just their mean, without hand-tuning personas.
2. **Macro closure.** Few LLM-agent economies impose general-equilibrium consistency or are estimated to data; linking LLM expectation modules to HANK or DSGE models (as Moll's critique invites) is largely untried.
3. **Clean evaluation.** Agreed protocols for post-cutoff testing, fake-date diagnostics and chronologically consistent models, so results are comparable across papers.
4. **Measurement-error correction.** Routine use of validation samples when LLM-generated series enter VARs and local projections.
5. **Scenario scoring.** Metrics for scenario quality (coverage of tail risks, internal consistency, usefulness to decision-makers) that do not rely on another LLM as judge.
6. **Lucas-critique robustness.** Whether LLM agents' behaviour shifts appropriately under genuinely new policy regimes, rather than reproducing training-data correlations.

## References

Linked items were opened for this review; unlinked items are cited from the literature reviews in Garavito-Acosta et al. (2026) and Eliseev & Seleznev (2026).

**Agents (Section 1)**

- Anesti, Hill & Joseph — [Inflation attitudes of large language models](https://www.bankofengland.co.uk/-/media/boe/files/working-paper/2026/inflation-attitudes-of-large-language-models.pdf), BoE SWP 1190 (2026)
- BIS ACC network — [Information treatments and household inflation expectations in Indonesia: a generative AI approach](https://www.bis.org/events/accconf2026/Paper_1_ID_251216.pdf) (2026)
- Bybee — [Surveying Generative AI's Economic Expectations](https://arxiv.org/pdf/2305.02823) (2023); [The Ghost in the Machine](https://conference.nber.org/conf_papers/f194779/f194779.pdf)
- *CFOs Meet LLMs* — [arXiv 2606.13812](https://arxiv.org/pdf/2606.13812) (2026)
- del Rio-Chanona, Pangallo & Hommes — [Can Generative AI agents behave like humans?](https://arxiv.org/abs/2505.07457v1) (2025)
- Feng et al. — [SimCity](https://arxiv.org/abs/2510.01297v1) (2025)
- Gusella & Vicario — [Generative Agents and Expectations](https://ideas.repec.org/p/arx/papers/2511.08604.html) (2025)
- Hansen & Lee — [Financial Stability Implications of Generative AI](https://www.federalreserve.gov/econres/feds/financial-stability-implications-of-generative-ai-taming-the-animal-spirits.htm), FEDS 2025-090
- Hansen, Horton, Kazinnik, Puzzello & Zarifhonarvar — [Simulating the Survey of Professional Forecasters](https://john-joseph-horton.com/papers/simulating-the-survey-of-professional-forecasters/)
- Hao & Xie — [Multi-LLM-Agent-Based framework](https://arxiv.org/html/2502.16879v1) (2025)
- Henning et al. — [LLM agents do not replicate human market traders](https://arxiv.org/pdf/2502.15800v3) (2025)
- Horton, Filippas & Manning — [Homo Silicus](https://arxiv.org/abs/2301.07543v2) (2023, rev. 2026)
- Kazinnik — [Bank Run, Interrupted](https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2024.06807), *Management Science*
- Kazinnik & Sinclair — [FOMC In Silico](https://digitaleconomy.stanford.edu/publication/fomc-in-silico-a-multi-agent-system-for-monetary-policy-decision-modeling); [AEA 2027 session](https://www.aeaweb.org/conference/2027/preliminary/1349)
- Li, Gao, Li, Li & Liao — [EconAgent](https://aclanthology.org/2024.acl-long.829), ACL 2024
- Lin, Sun & Yan — [Simulating Macroeconomic Expectations using LLM Agents](https://arxiv.org/abs/2505.17648v2) (2025)
- Moll — [The Trouble with Rational Expectations in Heterogeneous Agent Models](https://www.tse-fr.eu/seminars/2025-trouble-rational-expectations-heterogeneous-agent-models-challenge-macroeconomics) (2025)
- *Shachi* — [arXiv 2509.21862](https://arxiv.org/pdf/2509.21862) (2025)
- Tariff Threats, Macroeconomic Expectations, and Policy Communication — [arXiv 2608.30522](https://huggingface.co/papers/2608.30522) (2026)
- Wu, Xi & Xie — [LLM Survey Framework](https://www.econ.cuhk.edu.hk/wp-content/uploads/2025/11/XieShihan.pdf), NBER WP 34308 (2025)
- Zarifhonarvar — [Experimental evidence on LLM inflation expectations](https://marginalrevolution.com/marginalrevolution/2024/05/experimental-evidence-on-large-language-models.html) (2024); *Generating inflation expectations with LLMs*, *J. Monetary Economics* 157 (2026)

**Forecasts (Section 2)**

- Alam, Boyle, Li & Sekhposyan — *ChatMacro*, FRBSF WP 2026-04 ([abstract](https://www.aeaweb.org/conference/2027/preliminary/1349))
- André, Bessec & Goulby — [LLMs vs. econometric models for nowcasting GDP growth](https://isor.univie.ac.at/fileadmin/user_upload/i_isor/ISOR_Colloquium_Archives/Underlying_paper_Econometric_LLM.pdf) (2025)
- Bernasconi, Gómez García & Manzano Quiroga — [Ask, Think, Predict](https://aaep.org.ar/works/works2025/4806.pdf) (2025)
- Carriero, Pettenuzzo & Shekhar — [Macroeconomic Forecasting with Large Language Models](https://arxiv.org/pdf/2407.00890) (2024, rev. 2026)
- Dass, Racheter, Scheufele, Stucki, Vollmer & Winter — [Real-Time Macroeconomic Forecasting with TSFMs](https://www.bankofengland.co.uk/-/media/boe/files/events/2026/y-stucki-slides.pdf), SNB (2026)
- Faria-e-Castro & Leibovici — [Artificial Intelligence and Inflation Forecasts](https://www.stlouisfed.org/publications/review/2024/nov/artificial-intelligence-and-inflation-forecasts), St. Louis Fed Review (2024)
- Garavito-Acosta, Martínez-Rivera, Ospina-Tejeiro & Caicedo-García — [Experimenting with LLMs for Inflation Forecasting in Colombia](https://repec.graduateinstitute.ch/pdfs/Working_papers/HEIDWP23-2026.pdf) (2026)
- Giannellis et al., *J. Forecasting* (2026); Gonzalez (2026); Benny et al. (2025); Halawi et al., NeurIPS (2024); Karger et al., *ForecastBench*, ICLR (2025)
- Iadisernia & Camassa — [Prompting for Policy](https://ideas.repec.org/p/arx/papers/2511.02458.html), ICAIF (2025)
- Koyuncu, Kwon, Lombardi, Pérez-Cruz & Shin — [BISTRO](https://www.bis.org/publ/work1337.htm), BIS WP 1337 (2026)
- Massey University — [Generalisation Bounds of Zero-Shot Economic Forecasting using TSFMs](https://mro.massey.ac.nz/handle/10179/73995) (2025)
- Pham & Cunningham — [ChatGPT Can Predict the Future when it Tells Stories Set in the Future](https://arxiv.org/abs/2404.07396v2) (2024)

**Plug-in values (Section 3)**

- Allard, Teiletche & Zinebi — [Enhancing Inflation Nowcasting with LLM](https://arxiv.org/html/2410.20198v1) (2024)
- Aruoba & Drechsel — [Identifying Monetary Policy Shocks: A Natural Language Approach](https://nber.org/system/files/working_papers/w32417/w32417.pdf)
- Betz, Dix & Streit — [Identifying monetary policy shocks in newspapers using GPT](https://cepr.org/publications/DP21390), CEPR DP 21390
- de Bondt & Sun — [Enhancing GDP nowcasts with ChatGPT](https://www.suerf.org/publications/suerf-policy-notes-and-briefs/gdp-nowcasting-power-of-pmi-news-releases-using-chatgpt/), ECB WP 3063 (2025)
- Fernández-Fuertes — [Monetary Policy Shocks: A New Hope](https://econpapers.repec.org/paper/bafcbafwp/cbafwp25257.htm), Bocconi WP 25257 (2025)
- Hansen & Kazinnik — [Can ChatGPT Decipher Fedspeak?](https://www.frbsf.org/research-and-insights/publications/system-research-richmond-fed/2024/03/can-chatgpt-decipher-fedspeak/)
- Jha, Qian, Weber & Yang — [Generative AI, Managerial Expectations, and Economic Activity](https://www.arxiv.org/abs/2410.03897)
- Jaworski, *J. Forecasting* (2026); Tawichsri et al., PIER DP 252 (2026)
- Kermani et al. — [In the Fed's Mind](https://www.lse.ac.uk/economics/Assets/Documents/AMIR-Kermani-In-the-Feds-Mind.pdf) (2025)
- Kwon, Park, Rungcharoenkitkul & Smets — [Parsing the pulse](https://www.bis.org/publ/work1294.htm), BIS WP 1294 (2025)
- Yee & Sharma — [Calibrating Behavioral Parameters with LLMs](https://arxiv.org/abs/2602.01022v1) (2026)

**Scenarios (Section 4)**

- Badekale & Akinfaderin — [AI-Driven Climate Policy Scenario Generation for Sub-Saharan Africa](https://arxiv.org/abs/2505.18694v1) (2025)
- Collodel — [Interpreting the Interpreter](https://arxiv.org/html/2508.13635v2), Central Bank of Malta (2025)
- Ross & Ross — [AI-Simulated Expert Panels for Socio-Technical Scenarios](https://arxiv.org/pdf/2603.29470) (2026)
- Soleimani — [LLM-Generated Counterfactual Stress Scenarios](https://arxiv.org/abs/2512.07867) (2025)

**Methodology (Section 5)**

- Dunn, Meade, Sinha & Kabir — *Information Leakage in LLMs* ([abstract](https://www.aeaweb.org/conference/2027/preliminary/1349))
- Eliseev & Seleznev — [Fake Date Tests](https://www.cbr.ru/statichtml/file/188046/wp_167.pdf), Bank of Russia WP 167 (2026)
- Li, Chen, Namkoong & Peng — [LLM Generated Persona is a Promise with a Catch](https://arxiv.org/pdf/2503.16527) (2025)
- Lopez-Lira, Tang & Zhu — [The Memorization Problem](https://papers.ssrn.com/abstract=5217505) (2025, rev. 2026)
- Crane, Karra & Soto, FEDS 2025-044; He et al. (2025); Ludwig, Mullainathan & Rambachan, NBER WP 33344; Schoenegger et al. (2025); Bisbee et al., *Political Analysis* (2024); Paleka et al. (2025)

**Structural-model links (Section 3b)**

- Chatterjee & Kundu — [The AI Augmented Macro Economy](https://ir.vidyasagar.ac.in/bitstream/123456789/7958/1/A4.pdf), *Vidyasagar Univ. J. of Commerce* 30 (2025)
- Chib, Tan & Zhang — [Learning the Macroeconomic Language](https://arxiv.org/pdf/2512.21031) (2025, rev. 2026)
- Coibion, Gorodnichenko & Kamdar — [The Formation of Expectations, Inflation and the Phillips Curve](https://www.nber.org/papers/w23304.pdf), *JEL* (2018)
- Dubey, Gupta & Agarwal — [Grounding LLMs in DSGE Simulators for Policy Generation and Forecasting](https://arxiv.org/html/2610.01128) (2026)
- Wang & Liu — [Bridging Expectation Signals: a Behavioral Kalman Filter Framework](https://arxiv.org/pdf/2601.17527) (2026)
