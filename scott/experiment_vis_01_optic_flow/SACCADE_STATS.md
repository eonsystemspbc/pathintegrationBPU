# Free-flight saccade & intersaccadic flight statistics in flies
### Parameters for a biologically-faithful *Drosophila* optic-flow stimulus generator

**Compiled:** 2026-07-09 · **Scope:** body-yaw saccades, head/gaze saccades, gaze stabilization, translational speeds, roll/pitch banking, and the active-vision (rotation/translation-separation) claim. **Primary species:** *Drosophila melanogaster*; blowfly/hoverfly flagged as extrapolation throughout.

---

## 0. Critical framing (read first)

- **Flies have fixed compound eyes — there are no primate-style eye saccades.** "Saccade" here means the **body-yaw saccade during flight**: a brief, fast rotational turn of the body, punctuating longer **intersaccadic intervals** of relatively straight, translation-dominated flight.
- Flies additionally make **head/gaze saccades** (fast neck-driven head turns) and perform **gaze stabilization** — between saccades the head *counter-rotates* to reduce retinal image slip.
- **Body motion ≠ retinal (gaze) motion.** For a photoreceptor-input model, what the eye actually *sees* is the **gaze/retinal** trajectory, which is smoother than the body-yaw trajectory because of head stabilization. Both are reported below; the recommended-parameters section states an explicit body-vs-gaze modeling choice.
- **Two behavioral regimes, very different kinematics:** **spontaneous/cruising** saccades (voluntary, weakly visually gated) vs **evasive/escape** saccades (looming-evoked). Do not blend them.
- **Species caveat that recurs below:** the two highest-resolution *free-flight 3D body-kinematics* papers from the Dickinson lab (Muijres et al. 2014 escape; Muijres et al. 2015 spontaneous) used **_Drosophila hydei_** (larger: body length ~3–3.5 mm) rather than *D. melanogaster* (~2.5 mm). The detailed *melanogaster* angular-velocity numbers (Mongeau & Frye 2017; Bender & Dickinson 2006) are **magnetically tethered** (yaw-only, added pin inertia). *There is a genuine gap in high-resolution free-flight 3D body-saccade kinematics for D. melanogaster specifically.*

---

## 1. Parameter table (Drosophila primary; blowfly/other flagged)

Body length used for BL·s⁻¹ conversions: *D. melanogaster* ≈ 2.5 mm, *D. hydei* ≈ 3 mm (approximate).

| # | Quantity | Value ± spread | Species | Flight condition | Source |
|---|----------|----------------|---------|------------------|--------|
| **Body-yaw saccade kinematics** |
| 1 | Saccade **duration** | **< 100 ms** (~90° turn); ~130 ms for ~120° in uniform surround | *D. melanogaster* | free, cruising | Tammero & Dickinson 2002; Mronz & Lehmann 2008 |
| 1 | Saccade **duration** | **49 ± 18 ms** (N=44, ≈9 wingbeats) | *D. hydei* ⚠ | free, spontaneous | Muijres et al. 2015 |
| 1 | Saccade **duration** | spontaneous **101 ± 53 ms** (median 83); bar-fixation **77 ± 32 ms** | *D. melanogaster* | magnetic tether ⚠ | Mongeau & Frye 2017 |
| 1 | Saccade **duration (evasive)** | ~**50 ms** total maneuver; bank→counter-bank ≈ **25 ms (~5 wingbeats)** | *D. hydei* ⚠ | free, **EVASIVE** | Muijres et al. 2014; Dickinson & Muijres 2016 |
| 2 | Saccade **amplitude** | **~90° modal**; mean ~120° in stationary/uniform surround | *D. melanogaster* | free, cruising | Tammero & Dickinson 2002 |
| 2 | Saccade **amplitude** | **93 ± 27°**, broad ~20–180°, unimodal | *D. hydei* ⚠ | free, spontaneous | Muijres et al. 2015 |
| 2 | Saccade **amplitude** | spontaneous **64 ± 34°** (median 56); bar-fixation **33 ± 18°** | *D. melanogaster* | magnetic tether ⚠ | Mongeau & Frye 2017 |
| 2 | Saccade **amplitude (evasive)** | ~**90°** heading change | *D. hydei* ⚠ | free, **EVASIVE** | Muijres et al. 2014 |
| 3 | **Peak yaw angular velocity** | **> 1000 °/s** (general free-flight saccade peak) | *D. hydei/mel.* | free | Dickinson & Muijres 2016 (review) |
| 3 | Yaw angular velocity | mean **976 ± 457 °/s** (median 929) spontaneous; **567 ± 270** bar-fixation | *D. melanogaster* | magnetic tether ⚠ | Mongeau & Frye 2017 |
| 3 | Whole-body angular velocity | can exceed **~2000 °/s (~30 rad/s)** for a 90° turn in ~50 ms | *D. hydei* ⚠ | free | Fry et al. 2003; Muijres et al. 2015 |
| 3 | Peak yaw (tethered) | **~500 °/s** peak (control) — *slowed by magnet pin inertia* | *D. melanogaster* | magnetic tether ⚠ | Bender & Dickinson 2006 |
| 4 | **Saccade rate** (open cruise) | **1.37 s⁻¹** overall (⇒ mean ISI ~730 ms); internal floor **~0.4 s⁻¹**; up to **~4.4 s⁻¹** when strongly stimulated | *D. melanogaster* | free, open 2 m arena (88 flies, 6613 saccades, 4814 s) | Censi et al. 2013 |
| 4 | Saccade rate vs **clutter** | rate **increases** with higher background contrast / clutter / obstacles; ISI distribution ≈ scale-free (inverse-square) across landscape scale | *D. melanogaster* | free, structured vs uniform | Tammero & Dickinson 2002 |
| 4 | **Intersaccadic interval** | ~**0.7–1 s** typical open cruise (from rate above) | *D. melanogaster* | free, open | Censi et al. 2013 |
| 5 | **Intersaccadic straightness** | course approximately straight; body saccades account for **~80%** of all heading change (rotation confined to saccades) | *D. hydei/mel.* | free | Dickinson & Muijres 2016; Tammero & Dickinson 2002 |
| 5 | Residual intersaccadic **yaw rate (°/s)** | ⚠ **no verified free-flight numeric value in _D. melanogaster_** — qualitatively low ("clean" translational flow); blowfly proxy below | — | — | see gaps §6 |
| **Translational speeds & body attitude** |
| 6 | **Forward cruising speed** | **0.2–0.9 m/s** ("normal" v_air < 0.85 m/s) ⇒ ~**80–360 BL/s** | *D. melanogaster* | free (wind-tunnel VR, 284 trajectories) | Medici & Fry 2012 |
| 6 | Max/level forward speed | ~**2 m/s** cited as historical upper bound ⚠ *likely tethered/performance; primary source unverified* | *D. mel.* (?) | — | secondary; **treat as soft max** |
| 6 | **Sideslip (lateral) velocity** | not published as an absolute number; **actively minimized** during saccades (yaw torque cancels sideways drift) | *D. hydei* ⚠ | free | Muijres et al. 2015; Dickinson & Muijres 2016 |
| 6 | Forward-vs-sideways ratio | **qualitative only:** forward ≫ lateral; sideslip suppressed rather than a fixed ratio | — | free | Dickinson & Muijres 2016 |
| 7 | **Roll (bank) angle — cruising saccade** | ≈ **30°**, achieved in a couple of wingbeats (quick roll + counter-roll) | *D. hydei* ⚠ | free, cruising | Muijres et al. 2015; Dickinson & Muijres 2016 |
| 7 | **Roll angle — evasive** | **≥ 90°** ("roll on their sides, almost inverted") | *D. hydei* ⚠ | free, **EVASIVE** | Muijres et al. 2014 |
| 7 | Straight-flight roll | ≈ **0°**, haltere-stabilized; 90% of imposed roll corrected in **30 ± 7 ms** (~7 wingbeats), latency ~5 ms | *D. melanogaster* | free, magnetic perturbation | Beatus et al. 2015 |
| 7 | **Body pitch — hovering** | ≈ **45°** (measured 47.5° hover) | *D. melanogaster* | free | Medici & Fry 2012 |
| 7 | Body pitch vs airspeed | weakly linear, **~46°→56°** across 0.2–0.9 m/s (NOT a strong "nose-down to go fast" trend) | *D. melanogaster* | free | Medici & Fry 2012 |
| 7 | Pitch during perturbation | imposed ~20° drop corrected in ~60 ms (~15 wingbeats); reaction 13 ± 2 ms | *D. melanogaster* | free, magnetic perturbation | Ristroph et al. 2013 |
| 7 | Steering strategy | **bank-to-turn dominant** (30° bank → lateral force ≈ 50% of weight support), corrective **yaw largely passive** via translation-induced torque coupling; inertia- not friction-dominated | *D. hydei/mel.* | free | Fry et al. 2003; Dickinson & Muijres 2016; Karásek et al. 2018 |
| 7 | Wingbeat frequency (context) | ~**200 Hz** (period ~4–5 ms); 189 Hz hover (*hydei*) | *D. mel./hydei* | — | Dickinson & Muijres 2016 |
| **Head / gaze (retinal motion)** |
| 8 | **Head yaw range of motion** | anatomically limited to ≈ **±15°** | *D. melanogaster* | rigid-tether VR ⚠ | Cellini, Salem & Mongeau 2021 |
| 8 | **Head-saccade (reset) peak velocity** | **> 500 °/s** | *D. melanogaster* | rigid-tether VR ⚠ | Cellini, Salem & Mongeau 2021 |
| 8 | **Head-saccade (reset) duration** | ballistic reset ≈ **50 ms** (<5% of flight time); full excursion incl. return < 200 ms | *D. melanogaster* | rigid / magnetic tether ⚠ | Cellini, Salem & Mongeau 2021; Cellini & Mongeau 2022 |
| 8 | **Gaze stabilization** (yaw) | smooth head movement reduces retinal slip by **up to ~70%** between saccades | *D. melanogaster* | rigid-tether VR ⚠ | Cellini, Salem & Mongeau 2021 |
| 8 | Gaze stabilization (blowfly) | intersaccade head angular velocity ≈ **half** the thorax (0–100 °/s); **roll almost fully compensated** by head counter-roll | *Calliphora* ⚠ | free | van Hateren & Schilstra 1999 |
| 8 | **Head-vs-body timing** | flies do **NOT** show primate "head-leads-body": head & body saccades near-**simultaneous** (Drosophila; walking blowfly) or head starts slightly **later** but is faster & finishes earlier (flying blowfly), compressing the gaze shift | *D. mel.* / *Calliphora* ⚠ | free/tethered | Land 1973; Cellini et al. 2021; van Hateren & Schilstra 1999 |
| **Active-vision functional claim** |
| 9 | Saccadic strategy **separates rotational vs translational optic flow** → intersaccadic flow carries depth (motion parallax); read out by lobula-plate tangential cells (HSE, FD1, VCH, HS) during intersaccades | **Established in *Calliphora*/*Eristalis***; extrapolated to *Drosophila* | free / replayed natural flow / model | Egelhaaf et al. 2012; Kern et al. 2005; Geurten et al. 2010; Schwegmann et al. 2014 |
| 9 | Depth range & timing constraint | usable nearness range ~**2 m**; needs a **rotation-free window ≥ ~50–70 ms** for reliable depth (first few cm post-saccade unreliable); EMD nearness R² = 0.41 ± 0.14 (up to 0.7) | *Calliphora* ⚠ (model) | replayed natural flow | Schwegmann et al. 2014; Kern et al. 2005 |
| 9 | Time budget | **>80%** of flight time is translational (rotation confined to <~20%); flight = 9 prototypical movement modes, rotational saccades only ~3–4% of data | *Eristalis* / *Calliphora* ⚠ | free | Egelhaaf et al. 2012; Geurten et al. 2010 |

⚠ = species and/or condition differs from free-flying *D. melanogaster*; treat as extrapolation.

---

## 2. Regime comparison — spontaneous/cruising vs evasive/escape

| Quantity | Spontaneous / cruising | Evasive / escape |
|----------|------------------------|------------------|
| Duration | ~50–130 ms (free ~50 ms; tethered ~80–100 ms) | ~50 ms total; ~25 ms bank phase; body reorient in ~1 wingbeat (~5 ms) |
| Amplitude | broad 20–180°, **modal ~90°** (tethered smaller, ~55–65°) | ~90° |
| Peak yaw rate | **>1000 °/s** free (mean ~900–1000 °/s tethered) | **>1000 °/s**, highest; stability sacrificed |
| Roll (bank) | ~**30°** | **≥90°** (near-inverted) |
| Rate | ~1.0–1.4 s⁻¹ open cruise; floor ~0.4 s⁻¹; ↑ with clutter (up to ~4 s⁻¹) | one-shot, evoked per looming event |
| Intersaccade | straight, translational optic flow | banked roll + counter-roll transient |

**For an optic-flow *cruising* generator, parameterize on the spontaneous/cruising column.** Reserve the evasive column for a separate "escape" stimulus condition if needed.

---

## 3. Recommended stimulus parameters (Drosophila free-cruising flight)

Given the cross-verified data, a *Drosophila* free-cruising-flight optic-flow generator should use:

- **Saccade duration:** ~**50 ms** (use the free-flight *D. hydei* value 49 ± 18 ms, which is the best free-flight number; the ~80–130 ms figures are tethered or low-temporal-resolution and run long). Draw from a distribution, e.g. 50 ± 15 ms.
- **Saccade amplitude:** **broad, unimodal, modal ~90°**, sampled roughly 20–180° (e.g. 90 ± 30°). Sign (left/right) set by scene asymmetry; magnitude is feed-forward at onset.
- **Peak yaw rate:** **~1500–2000 °/s** peak (consistent with a 90° turn in ~50 ms and the >1000 °/s free-flight figure); mean-over-saccade ~1000 °/s.
- **Saccade rate / intersaccadic interval:** **~1 saccade·s⁻¹ in open cruise** ⇒ **ISI ≈ 700–1000 ms**; add an internal floor near 0.4 s⁻¹ and *raise the rate in cluttered/near-obstacle conditions* (Tammero & Dickinson 2002; up to ~4 s⁻¹ when strongly stimulated).
- **Intersaccadic straightness:** hold near-**straight** flight between saccades so translational flow dominates; residual body yaw should be **low** (rotation contributes <~20% of heading change / time). Exact residual °/s for *D. melanogaster* is not published — if a value is needed, use the blowfly intersaccadic proxy (head yaw 0–100 °/s, roll near-fully compensated) and label it as such.
- **Translational speed:** **forward ~0.5 m/s** (range 0.2–0.9 m/s); **sideslip small** (actively suppressed — no published ratio; treat lateral ≪ forward, e.g. <10–20% and transient).
- **Roll (bank):** transient **~30° bank** locked to each cruising saccade (roll → counter-roll), returning to ~0° between saccades; reserve **≥90°** banks for an evasive condition only.
- **Pitch:** body pitch **~45–55°**, weakly increasing with forward speed; near-constant between saccades.

**Body-vs-gaze modeling choice (important):** For a *photoreceptor / optic-lobe input* model, drive the retina with the **gaze (head-stabilized) trajectory, not the raw body trajectory.** Concretely:
1. Generate the **body** trajectory with the parameters above.
2. Between saccades, **attenuate the body-induced rotational retinal slip**: multiply yaw slip by ~**0.3** (Drosophila ~70% reduction; use ~0.5 if adopting the blowfly figure) and set **roll slip ≈ 0** (roll is near-fully compensated by head counter-roll).
3. During saccades, inject a **brief (~50 ms) high-velocity (>500 °/s Drosophila; up to thousands °/s if blowfly-scaled) gaze-shift transient**, temporally **compressed** relative to the body turn (head is faster; do **not** advance it ahead of the body — flies are not "head-leads-body").
4. Translational flow is passed through essentially unattenuated. Net effect: the network sees **near-rotation-free, depth-carrying translational flow during the long intersaccadic fixations**, punctuated by short rotational bursts — exactly the active-vision structure that (in blowfly/hoverfly) segregates rotation from translation for depth-from-parallax.

If a simpler model is required, driving the retina directly with the **body** trajectory is a defensible first approximation, but it will overstate intersaccadic rotational flow (especially roll) and understate the cleanliness of the translational-depth signal — flag this explicitly.

---

## 4. The functional (active-vision) claim — state of evidence

**Claim:** the saccade-and-fixate flight strategy *temporally separates rotational and translational optic flow*, so that during straight intersaccadic intervals the retinal flow is translation-dominated and therefore carries **depth / motion-parallax** information (retinal velocity ∝ self-velocity / object-distance during translation; depth-independent during rotation).

**Verdict:** **well established and quantitatively supported — but almost entirely in larger flies** (blowfly *Calliphora*, hoverfly *Eristalis*), via a converging chain: behavioral kinematics (Schilstra & van Hateren 1999; van Hateren & Schilstra 1999), closed-loop/pathway modeling (Lindemann et al. 2005, 2008; Schwegmann et al. 2014), and neural recordings showing lobula-plate tangential cells (HSE, FD1, VCH, HS) encode scene nearness specifically during intersaccades (Kern et al. 2005; Liang et al. 2012). Egelhaaf et al. 2012 is the best single review citation.

**In *Drosophila*:** the *behavioral premise* is solid — saccadic heading changes, rotation squeezed into fast turns, ~80% of heading change in discrete saccades, and active head gaze-stabilization (Tammero & Dickinson 2002; Censi et al. 2013; van Breugel & Dickinson 2012; Cellini et al. 2021). But the **neural depth-from-intersaccadic-parallax readout has not been demonstrated in *Drosophila*** the way it has in blowfly. Treat *Drosophila* depth-from-parallax as a **well-motivated hypothesis / extrapolation, not a proven result** — a point worth stating explicitly in the experiment writeup, since this project models the *Drosophila* optic lobe.

---

## 5. Reference list (labnotebook-ready)

**_Drosophila melanogaster_ (primary species):**

1. Tammero, L.F. & Dickinson, M.H. (2002). The influence of visual landscape on the free flight behavior of the fruit fly *Drosophila melanogaster*. *Journal of Experimental Biology* **205**(3): 327–343. DOI: 10.1242/jeb.205.3.327.
2. Bender, J.A. & Dickinson, M.H. (2006). A comparison of visual and haltere-mediated feedback in the control of body saccades in *Drosophila melanogaster*. *Journal of Experimental Biology* **209**(23): 4597–4606. DOI: 10.1242/jeb.02583. [magnetically tethered] (companion: Bender & Dickinson 2006, *J Exp Biol* **209**(16): 3170–3182, DOI: 10.1242/jeb.02369.)
3. Censi, A., Straw, A.D., Sayaman, R.W., Murray, R.M. & Dickinson, M.H. (2013). Discriminating external and internal causes for heading changes in freely flying *Drosophila*. *PLoS Computational Biology* **9**(2): e1002891. DOI: 10.1371/journal.pcbi.1002891.
4. Fry, S.N., Sayaman, R. & Dickinson, M.H. (2003). The aerodynamics of free-flight maneuvers in *Drosophila*. *Science* **300**(5618): 495–498. DOI: 10.1126/science.1081944.
5. Medici, V. & Fry, S.N. (2012). Embodied linearity of speed control in *Drosophila melanogaster*. *Journal of the Royal Society Interface* **9**(77): 3260–3267. DOI: 10.1098/rsif.2012.0527.
6. Ristroph, L., Ristroph, G., Morozova, S., Bergou, A.J., Chang, S., Guckenheimer, J., Wang, Z.J. & Cohen, I. (2013). Active and passive stabilization of body pitch in insect flight. *Journal of the Royal Society Interface* **10**(85): 20130237. DOI: 10.1098/rsif.2013.0237.
7. Beatus, T., Guckenheimer, J.M. & Cohen, I. (2015). Controlling roll perturbations in fruit flies. *Journal of the Royal Society Interface* **12**(105): 20150075. DOI: 10.1098/rsif.2015.0075.
8. Mronz, M. & Lehmann, F.-O. (2008). The free-flight response of *Drosophila* to motion of the visual environment. *Journal of Experimental Biology* **211**(13): 2026–2045. DOI: 10.1242/jeb.008268.
9. van Breugel, F. & Dickinson, M.H. (2012). The visual control of landing and obstacle avoidance in the fruit fly *Drosophila melanogaster*. *Journal of Experimental Biology* **215**(11): 1783–1798. DOI: 10.1242/jeb.066498.
10. Mongeau, J.-M. & Frye, M.A. (2017). *Drosophila* spatio-temporally integrates visual signals to control saccades. *Current Biology* **27**(19): 2901–2914. DOI: 10.1016/j.cub.2017.08.035. [magnetically tethered]
11. Cellini, B., Salem, W. & Mongeau, J.-M. (2021). Mechanisms of punctuated vision in fly flight. *Current Biology* **31**(18): 4009–4024.e3. DOI: 10.1016/j.cub.2021.06.080. [rigid-tether VR — key Drosophila head/gaze paper]
12. Cellini, B. & Mongeau, J.-M. (2022). Nested mechanosensory feedback actively damps visually guided head movements in *Drosophila*. *eLife* **11**: e80880. DOI: 10.7554/eLife.80880.
13. Salem, W., Cellini, B., Frye, M.A. & Mongeau, J.-M. (2020). Fly eyes are not still: a motion illusion in *Drosophila* flight supports parallel visual processing. *Journal of Experimental Biology* **223**(10): jeb212316. DOI: 10.1242/jeb.212316.
14. Davis, B.A. & Mongeau, J.-M. (2023). The influence of saccades on yaw gaze stabilization in fly flight. *PLOS Computational Biology* **19**(12): e1011746. DOI: 10.1371/journal.pcbi.1011746. [control model]
15. Yang, H., Barredo, J., Currea, J.P., Sondhi, Y., Palavalli-Nettimi, R., Sponberg, S., Tarokh, V. & Theobald, J. (2024). Body size and light environment modulate flight speed and saccadic behavior in free-flying *Drosophila melanogaster*. bioRxiv 2024.07.08.602594. DOI: 10.1101/2024.07.08.602594. [preprint]

**_Drosophila hydei_ (larger congener — free-flight 3D kinematics; flag as extrapolation to *melanogaster*):**

16. Muijres, F.T., Elzinga, M.J., Melis, J.M. & Dickinson, M.H. (2014). Flies evade looming targets by executing rapid visually directed banked turns. *Science* **344**(6180): 172–177. DOI: 10.1126/science.1248955. [**EVASIVE/escape**]
17. Muijres, F.T., Elzinga, M.J., Iwasaki, N.A. & Dickinson, M.H. (2015). Body saccades of *Drosophila* consist of stereotyped banked turns. *Journal of Experimental Biology* **218**(6): 864–875. DOI: 10.1242/jeb.114280. [spontaneous]
18. Karásek, M., Muijres, F.T., De Wagter, C., Remes, B.D.W. & de Croon, G.C.H.E. (2018). A tailless aerial robotic flapper reveals that flies use torque coupling in rapid banked turns. *Science* **361**(6407): 1089–1094. DOI: 10.1126/science.aat0350. [robotic model of *D. hydei* escape]

**Reviews / cross-species (Drosophila + blowfly):**

19. Dickinson, M.H. & Muijres, F.T. (2016). The aerodynamics and control of free flight manoeuvres in *Drosophila*. *Philosophical Transactions of the Royal Society B* **371**(1704): 20150388. DOI: 10.1098/rstb.2015.0388.
20. Land, M.F. (1973). Head movement of flies during visually guided flight. *Nature* **243**: 299–300. DOI: 10.1038/243299a0.
21. Land, M.F. (1999). Motion and vision: why animals move their eyes. *Journal of Comparative Physiology A* **185**: 341–352. DOI: 10.1007/s003590050393.
22. Egelhaaf, M., Boeddeker, N., Kern, R., Kurtz, R. & Lindemann, J.P. (2012). Spatial vision in insects is facilitated by shaping the dynamics of visual input through behavioral action. *Frontiers in Neural Circuits* **6**: 108. DOI: 10.3389/fncir.2012.00108. [best single active-vision review]

**Blowfly (_Calliphora_) & hoverfly (_Eristalis_) — larger/faster; flag as extrapolation:**

23. Schilstra, C. & van Hateren, J.H. (1999). Blowfly flight and optic flow. I. Thorax kinematics and flight dynamics. *Journal of Experimental Biology* **202**(11): 1481–1490. DOI: 10.1242/jeb.202.11.1481. [*Calliphora vicina*, free flight]
24. van Hateren, J.H. & Schilstra, C. (1999). Blowfly flight and optic flow. II. Head movements during flight. *Journal of Experimental Biology* **202**(11): 1491–1500. DOI: 10.1242/jeb.202.11.1491. [*Calliphora vicina* — primary head-kinematics source]
25. Schilstra, C. & van Hateren, J.H. (1998). Stabilizing gaze in flying blowflies. *Nature* **395**(6703): 654. DOI: 10.1038/27114.
26. Kern, R., van Hateren, J.H., Michaelis, C., Lindemann, J.P. & Egelhaaf, M. (2005). Function of a fly motion-sensitive neuron matches eye movements during free flight. *PLoS Biology* **3**(6): e171. DOI: 10.1371/journal.pbio.0030171.
27. Lindemann, J.P., Kern, R., van Hateren, J.H., Ritter, H. & Egelhaaf, M. (2005). On the computations analyzing natural optic flow: quantitative model analysis of the blowfly motion vision pathway. *Journal of Neuroscience* **25**(27): 6435–6448. DOI: 10.1523/JNEUROSCI.1132-05.2005.
28. Lindemann, J.P., Weiss, H., Möller, R. & Egelhaaf, M. (2008). Saccadic flight strategy facilitates collision avoidance: closed-loop performance of a cyberfly. *Biological Cybernetics* **98**(3): 213–227. DOI: 10.1007/s00422-007-0205-x.
29. Geurten, B.R.H., Kern, R., Braun, E. & Egelhaaf, M. (2010). A syntax of hoverfly flight prototypes. *Journal of Experimental Biology* **213**(14): 2461–2475. DOI: 10.1242/jeb.036079. [*Eristalis tenax*]
30. Liang, P., Heitwerth, J., Kern, R., Kurtz, R. & Egelhaaf, M. (2012). Object representation and distance encoding in three-dimensional environments by a neural circuit in the visual system of the blowfly. *Journal of Neurophysiology* **107**(12): 3446–3457. DOI: 10.1152/jn.00530.2011.
31. Schwegmann, A., Lindemann, J.P. & Egelhaaf, M. (2014). Depth information in natural environments derived from optic flow by insect motion detection system: a model analysis. *Frontiers in Computational Neuroscience* **8**: 83. DOI: 10.3389/fncom.2014.00083.
32. Kress, D. & Egelhaaf, M. (2004). Saccadic head and thorax movements in freely walking blowflies. *Journal of Comparative Physiology A* **190**. DOI: 10.1007/s00359-004-0541-4.

**Peripheral (finest retinal-motion layer, if ever needed):**

33. Fenk, L.M., Avritzer, S.C., Weisman, J.L., Nair, A., Randt, L.D., Mohren, T.L., Siwanowicz, I. & Maimon, G. (2022). Muscles that move the retina augment compound-eye vision in *Drosophila*. *Nature* **610**: 116–122. DOI: 10.1038/s41586-022-05317-5. [~2–3° retinal-muscle movements, distinct from head saccades]

---

## 6. Gaps & uncertainties (explicit)

1. **Residual intersaccadic yaw rate (°/s), free-flight _D. melanogaster_ — WEAKEST QUANTITY.** Qualitatively established (straight segments, translational optic flow, ~80% of heading change in saccades) but **no verified primary numeric value**. The quantitative work is either tethered or modeling (Davis & Mongeau 2023). Blowfly proxy: intersaccade head yaw 0–100 °/s, roll near-fully compensated. If a hard number is required, use the blowfly proxy and label it, or reanalyze raw *melanogaster* trajectories.
2. **Species mismatch on the best free-flight kinematics.** Modal ~90° amplitude, ~50 ms duration, ~30° bank, >1000 °/s peak all come from **_D. hydei_** (Muijres 2014/2015). The corresponding *D. melanogaster* free-flight 3D body-rotation kinematics do not exist at the same resolution; *melanogaster* angular-velocity numbers are tethered. BL/s conversions are approximate.
3. **Cluttered-vs-open ISI, quantified in _D. melanogaster_ free flight.** Direction of effect is clear (clutter/contrast ↑ saccade rate; Tammero 2002) but a clean paired ISI number for cluttered vs open is not well pinned.
4. **Absolute sideslip velocity and a clean forward:sideways ratio** — not published as numbers; literature states sideslip is *minimized*, not a fixed fraction.
5. **Max forward speed ~2 m/s** — appears only in secondary sources as a historical/likely-tethered figure; trustworthy free-flight cruising range is **0.2–0.9 m/s** (Medici & Fry 2012). Treat 2 m/s as a soft upper bound.
6. **Pitch-vs-speed direction** — weak, near-constant ~45–56°; do NOT overstate a "nose-down with speed" trend.
7. **Clean _Drosophila_ head-saccade amplitude ± SD / duration table** — thin in open sources; best verified values are head range ±~15°, reset peak >500 °/s, reset ~50 ms, retinal-slip reduction up to ~70% (all from tethered preps — Cellini et al. 2021/2022). Body-free vs body-fixed differences are figure-only.
8. **Head-leads-body lag (ms)** — NOT applicable to flies; head/body saccades are near-simultaneous (Drosophila) or head slightly lags then compresses the shift (blowfly). Do not model a head lead.
9. **Depth-from-intersaccadic-parallax neural readout** — demonstrated in **blowfly/hoverfly**, not in *Drosophila*. Extrapolation to the *Drosophila* optic lobe is well-motivated but unproven; state this caveat in the experiment.

---
*No git commit made. No files under `scripts/flow/` touched. Numbers are cited; anything inferred is labeled.*
