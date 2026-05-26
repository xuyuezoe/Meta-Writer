---
paper_id: d2fcbc99be3a47429e8057dbbee5eb3e
doi: 10.1101/2023.12.13.571122
source: biorxiv
version_date: '2025-12-11'
license: null
title: Functional imaging of nine distinct neuronal populations under a miniscope in freely behaving animals
authors:
- name: Mary L. Phillips
  affiliations:
  - 1
  - 2
  corresponding: true
  email: mary.phillips@zeiss.com
- name: Nicolai T. Urban
  affiliations:
  - 1
  corresponding: false
  email: null
- name: Taddeo Salemi
  affiliations:
  - 1
  corresponding: false
  email: null
- name: Zhe Dong
  affiliations:
  - 3
  corresponding: false
  email: null
- name: Ryohei Yasuda
  affiliations:
  - 1
  corresponding: true
  email: ryohei.yasuda@mpfi.org
affiliations:
  1: Max Planck Florida Institute for Neuroscience, Jupiter, FL.
  2: ZEISS Research Microscopy Solutions, White Plains, NY.
  3: MetaCell, Boston, MA.
abstract: Head-mounted miniscopes have enabled functional fluorescence imaging in freely moving animals. However, current
  technology is limited to recording at most two spectrally distinct fluorophores, severely restricting the number of identifiable
  cell types. Here we introduce multiplexed neuronal imaging (Neuroplex), a pipeline combining miniscope Ca$^{2+}$ recordings
  with in vivo multiplexed confocal spectral imaging to distinguish nine projection-defined neuronal subtypes through the
  same GRIN lens. By co-registering defined neurons with fluorophore-specific spectral fingerprints via linear unmixing, we
  link projection-defined identities to behaviorally relevant neuronal activity. This approach overcomes spectral constraints
  of miniscopes, enabling circuit-level dissection of behavior in single animals.
keywords: null
paper_type: methods
subject_areas:
- Neuroscience
- Optical Imaging
- Biotechnology
- Systems Neuroscience
datasets: null
stats:
  word_count: 12168
  has_math: true
  section_count: 58
---
# Functional imaging of nine distinct neuronal populations under a miniscope in freely behaving animals

# Functional imaging of nine distinct neuronal populations under a miniscope in freely behaving animals  

Mary L. Phillips$^{1,2,*}$, Nicolai T. Urban$^{1}$, Taddeo Salemi$^{1}$, Zhe Dong$^{3}$, Ryohei Yasuda$^{1,*}$  

$^{1}$Max Planck Florida Institute for Neuroscience, Jupiter, FL.  

$^{2}$ZEISS Research Microscopy Solutions, White Plains, NY.  

$^{3}$MetaCell, Boston, MA.  

*Correspondence: mary.phillips@zeiss.com, ryohei.yasuda@mpfi.org  

# Abstract  

Head-mounted miniscopes have enabled functional fluorescence imaging in freely moving animals. However, current technology is limited to recording at most two spectrally distinct fluorophores, severely restricting the number of identifiable cell types. Here we introduce multiplexed neuronal imaging (Neuroplex), a pipeline combining miniscope Ca$^{2+}$ recordings with in vivo multiplexed confocal spectral imaging to distinguish nine projection-defined neuronal subtypes through the same GRIN lens. By co-registering defined neurons with fluorophore-specific spectral fingerprints via linear unmixing, we link projection-defined identities to behaviorally relevant neuronal activity. This approach overcomes spectral constraints of miniscopes, enabling circuit-level dissection of behavior in single animals.  

# Main  

A central goal of systems neuroscience is to understand how cell-type-specific activity gives rise to behavior. Miniaturized head-mounted microscopes have enabled in vivo calcium imaging in freely moving animals, greatly enhancing our understanding of neural encoding of behavior and experience$^{1}$. In the past decade these "miniscopes"$^{2}$ have evolved rapidly, gaining features such as wireless capability$^{3}$, increased field of view$^{4,5}$, two-photon capability$^{6-8}$, multiple focal planes$^{9}$, and the ability to optogenetically stimulate as well as record$^{10}$. Despite these advancements, the size and weight constraints of head-mountable microscopes continue to limit the amount of information obtainable from any single experiment.  

One major constraint is spectral capacity: most miniscopes can distinguish only one or two fluorophores. Physical limitations on internal optics and filter sets restrict excitation and emission capabilities. In addition, the gradient-index (GRN) lens induces severe chromatic aberrations along the optical z-axis, shifting the focal plane based on wavelength, often beyond the miniscope's focusing range$^{11}$. As a result, the current state-of-the-art typically allows the use of GCaMP in the green channel and a second marker in the red wavelength domain$^{11}$  

To analyze more than two neuronal subtypes, researchers have resorted to using different animals with replicable behavioral tasks across cohorts$^{12,13}$. While this approach can yield meaningful comparisons, it prevents simultaneous analysis of distinct cell types within the same circuit. Several groups have attempted to overcome this limitation using post-hoc immunohistochemistry or in situ profiling, but these approaches are labor-intensive and incompatible with longitudinal studies$^{14-17}$.  

To address these challenges, we developed Neuroplex, a flexible imaging and analysis pipeline that enables the identification of multiple projection-defined neuronal subtypes observed during in vivo calcium imaging in freely behaving animals. Neuroplex combines functional recordings acquired through a head-mounted miniscope with multiplexed spectral confocal imaging performed through the same implanted GRIN lens, allowing fluorophore identities to be assigned to functionally defined neurons within the same, live animal. This is achieved by first registering the location of active neurons measured with the miniscope to the confocal image stack using a Python-based image registration workflow and then extracting multiplexed spectral data from those registered neuronal locations. A linear  

unmixing algorithm based on experimentally derived spectral fingerprints is applied to assign one or more fluorophore identities to each registered neuronal location.  

We demonstrate this approach using nine spectrally distinct fluorophores delivered via retrograde AAVs to downstream targets of the medial prefrontal cortex (mPFC), enabling the identification of projection-defined pyramidal neuron populations in vivo. Across animals, approximately 70% of functionally defined neurons could be assigned to one of the injected fluorophores, with minimal false positives and high robustness supported by simulation and experimental modeling. Because all imaging is performed through the implanted GRIN lens in vivo, Neuroplex is fully compatible with longitudinal designs, allowing users to assess fluorophore identity prior to behavioral testing, or to re-image the same cell populations across multiple time points. This method avoids the challenges inherent to post-hoc co-registration between in vivo and post-fixation images, and expands the utility of miniscope-based experiments for circuit-level dissection.  

C) Process functional data  

# Fig. 1: Experimental pipeline for identifying 9 neuronal subtypes within behaviorally relevant ROIs determined by GCaMP6s imaging.  

a, Surgical paradigm. In a TetO-GCaMP6s x CaMKII-tTa mouse, 9 AAV$_{retro}$ viruses are injected into downstream brain regions and GRIN lens implanted into the target region. b, Simultaneous recording of GCaMP6s (top) and behavior (bottom) during a social memory task. Scale bar = 100 µm c, GCaMP6s recordings are processed. CNMF-defined ROIs (top) and ΔF/F traces (bottom) are exported. Scale bar = 100 µm. d, Mice are head fixed and FOV under the GRIN lens imaged using the multiplexed lambda method. e, Transformations are determined using anatomical background images to co-register the two imaging platforms. The transformations are applied to CNMF-defined ROIs. Scale bar = 100 µm. f, Multispectral data are collected for each ROI (top) and an average spectral fingerprint for all ROIs is generated (bottom). Mean ± 1.5 SD. Scale bar = 100 µm. g, A linear unmixing model is applied to determine the fluorophore contribution for each ROI. Scale bar = 100 µm. i, Neural activity is sorted by cell type. Scale bars = 20 ΔF/F (vertical), 20 seconds (horizontal).  

# Results  

# Correcting for GRIN-induced chromatic aberration  

Neuroplex enables simultaneous tracking of multiple neuronal subtypes in behaving animals through a three-step pipeline: (1) Head-mounted GRIN-lens based miniscope imaging of GCaMP activity during behavior; (2) in vivo multiplexed confocal spectral imaging through the same GRIN lens to capture fluorophore fingerprints; and (3) linear unmixing to assign projection-specific identities to functionally defined neurons (Fig. 1). Since the system is strongly influenced by the optical characteristics of the GRIN lens, we first characterized the chromatic and geometric aberrations of the most commonly used lens types in the field: $1 \times 4$ mm silver-doped lenses, typically used for mPFC imaging, and $0.6 \times 7$ mm lenses, available in both silver-doped and lithium-doped glass. The $1 \times 4$ mm format is currently only manufactured in silver-doped glass, which has a higher numerical aperture (NA). Lithium-doped lenses are more frequently used in dual-color experiments due to their reduced chromatic dispersion. Multiple lenses of each type were tested, and inter-lens variability was found to be negligible.  

GRIN lenses introduce optical aberrations, including axial chromatic shifts$^{19}$ and lateral astigmatism. We measured the optical aberrations of each GRIN lens using a fluorescent calibration slide containing a pattern of equally spaced rings (each 1.8 μm in diameter, spaced 50 μm apart) that were both excitable and emitted throughout the entire visible spectrum$^{20}$. A custom-made GRIN lens holder enabled precise and repeatable positioning above the calibration slide. We observed substantial chromatic aberrations along the optical z-axis, which increased in severity with the length of the GRIN lens (Fig. 2). These aberrations caused a downward shift of the focal plane at longer wavelengths, which followed a second-order polynomial (Fig. 2a, b and Supp. Fig. 1a). The axial chromatic shift was significantly smaller for lithium-doped lenses as opposed to silver-doped lenses (Supp. Fig. 1b). Chromatic aberrations in the lateral (x, y) plane were negligible across the field of view, including at the periphery (Fig. 2c), indicating minimal off-axis distortion.  

To overcome the axial chromatic aberrations during in vivo imaging, we adopted a two-step corrective strategy. First, we acquired spectral z-stacks spanning the full focal depth of the visible spectrum, using 405 nm excitation with nm emission to define the upper bound and nm excitation with nm emission for the lower bound. These z-stacks were subsequently flattened to remove any wavelength-dependent focal shifts. Second, we widened the confocal pinhole to 350 µm to retain the longer-wavelength emission that would otherwise be excluded due to chromatic displacement from the nominal focal plane.  

In addition to the axial chromatic aberrations, light transmission through GRIN lenses is also strongly wavelength dependent. Transmission peaked at 87% between 550-600 nm but declined sharply below 500 nm, falling to 58% at nm (Fig. 2d). This effect is most pronounced in silver-doped lenses and increased with lens length (Supp. Fig. 1c, d). We fit a $6^{th}$-order polynomial to the transmission spectra to model wavelength-dependent scattering and used the resulting Phillips et al. Multicolor neuron identification within miniscope 3  

function to adjust excitation laser powers during confocal imaging: for in vivo experiments, we first identified which excitation laser induced the brightest emitted spectral bin. We then adjusted that laser's power and detector gain to achieve full dynamic range without saturation. The powers of all other lasers were subsequently scaled relative to this reference, using the polynomial-based transmission correction to ensure uniform illumination across wavelengths.  

In addition to chromatic distortions, the GRIN lenses also induced lateral aberrations, particularly in the outer third of the FOV. We did not observe any directional field distortions (Fig. 2e), but instead very prominent rotationally symmetric astigmatism as well as a strong curvature of the focal plane, both growing more severe with increasing distance from the center of the lens (Fig. 2f, g). Despite their severity, these distortions were largely achromatic (Fig. 2c), thus not requiring any chromatic corrections in the xy-plane.  

a)  

c)  

b)  

d)  

# Fig. 2: GRIN lens induced chromatic aberrations  

a, Multicolor image obtained through 1x4 mm silver-doped GRIN lens of the calibration slide highlighting the z-plane chromatic aberration. b, Shift in z-focal plane as a function of excitation laser wavelength. Second-order polynomial R² = 0.9926, n = 5. c, Orthogonal projection of multicolor image obtained through 1x4mm silver-doped GRIN lens of the calibration slide. Intensity profile (below) of single ring for each excitation channel shows negligible chromatic shift along lateral axes. d, Percent transmission through the GRIN lens as a function of excitation laser wavelength. Sixth order polynomial R² = 0.9751, n = 5. e, Orthogonal projection of calibration slide imaged through 1x4mm silver-doped GRIN lens overlaid (in cyan) with rectilinear grid lines. Substantial overlap of fluorescent rings from the grid indicates minimal field distortions. f, Excerpt from (e) showing the rings focused in the sagittal plane (z= +20 µm), the circle of least confusion (z= 0 µm), and the tangential focal plane (z= -17.5 µm). g, Curvature of the Petzval field as a function of radial distance from center of the GRIN lens. Astigmatism results in three axially separated focal planes. Second order polynomial sagittal R² = 0.9845, least confusion R² = 0.8839, and tangential R² = 0.7519, n = 3. Scale bars = 100 µm.  

# Fluorophore selection and optimization  

Careful fluorophore selection is critical when spectrally distinguishing many different fluorophores. We analyzed published spectral profiles of available genetically encoded fluorophores and identified a combination of fluorescent proteins that could be uniquely separated by considering both their excitation and their emission spectra$^{18}$. We determined that more fluorophores could be distinguished by employing a multiplexed-spectral imaging approach in which excitation lasers are sequentially activated and full emission spectra captured via spectral detectors. We then simulated the multiplexed spectral strategy in silico using various fluorophore combinations, before selecting ten fluorophores in addition to GCaMP6s for experimental validation. These included: mTagBFP2, mTurquoise2, T-Sapphire, mVenus, mPapaya, mOrange2, mScarlet, FusionRed, mCyRFP1, and mNeptune2.5. Each fluorophore was transfected individually into HEK293T cells, and its spectral fingerprint was recorded under multiplexed spectral imaging conditions for later use in linear unmixing (Supp. Fig. 2). mPapaya was found to induce marked cell death in HEK293T cells and was therefore excluded from further study. The nine remaining fluorophores were used to make retrograde adeno-associated viruses (AAV$_{\text{retro}}$) for in vivo application.  

# Identification of behaviorally relevant neurons  

To test whether we could detect ten distinct fluorophores through GRIN lenses in behaviorally relevant neurons, we targeted mPFC projection neurons labeled via retrograde transport from up to nine downstream brain regions. Mice stably expressing GCaMP6s in pyramidal neurons$^{21}$ were injected with the nine different AAV$_{\text{retro}}$ viruses, each encoding a unique fluorophore, following two different fluorophore-region maps. Each map assigned a fluorophore to a brain region known to receive projections from the mPFC. These regions included the dorsal periaqueductal gray (dPAG), basolateral amygdala (BLA), claustrum (Cla), nucleus accumbens (NAc), striatum (Str), locus coeruleus (LC), ventral tegmental area (VTA), lateral habenula (LHb), lateral hypothalamus (IHyp), and the contralateral prefrontal cortex (c-mPFC). AAV$_{\text{retro}}$ viruses are taken up by axon terminals and the encoded genetic sequences are transported back to the nucleus where the fluorophore is synthesized$^{22}$.  

During the same surgery, a 1x4 mm silver-doped GRIN lens with an integrated baseplate and head-bar was implanted directly above the prelimbic region of the mPFC (Fig. 1a). After allowing five weeks for recovery and fluorophore expression, mice underwent a social memory task while GCaMP6s activity was recorded using a head-mounted miniscope (Fig. 1b). The location of neurons active during this task was identified as regions of interest (ROIs) using a constrained non-negative matrix factorization algorithm (CNMF)$^{23}$. The number of these behaviorally relevant ROIs ranged from 105–440 between subjects. These ROIs, along with a time-averaged fluorescence image of the field of view, were exported for co-registration and spectral analysis (Fig. 1c).  

Following functional imaging of GCaMP6, we performed spectral imaging using a confocal microscope to identify fluorophore identity. To minimize the time-dependent changes of GCaMP emission during imaging, we silenced cortical activity by anesthetizing the animal with ketamine before head-fixing it under the confocal microscope to prevent motion. Using the parameters determined from GRIN lens characterization, multiplexed spectral z-stack images were acquired for the entire field of view under the GRIN lens (Fig. 1d, e, 3a). A rolling ball background subtraction of 30 µm was applied to each z-plane to reduce neuropil interference, and the stack was then summed along the z-axis to counteract chromatic z-aberration. To co-register the two imaging modalities, we input the nm emission channel from the nm excitation laser, which provides strong vascular contrast and therefore serves as the confocal reference  

a)  

b)  

c)  

e)  

f)  

d)  

# Fig. 3: Identification of 9 fluorophores through GRIN lenses in vivo  

a, In vivo multiplexed spectral imaging paradigm. Schematic of multiplexed spectral imaging (left). Depiction of overlapping fluorophore spectral emissions for each excitation laser wavelength (middle). Depiction of multiplexed spectral images which create a 204-dimensional dataset (right). b, Automated co-registration of miniscope and laser scanning confocal microscope (LSM) images. Top: A calibration slide used to measure scaling between modalities. Bottom: Experimental FOV showing brain vasculature. Miniscope and confocal images of the same FOV and automated co-registration overlay with zoomed-in regions of interest. c, Example calcium-activity ROI derived from miniscope data co-registered and overlaid on confocal LSM image. d, Spectral fingerprint of the example ROI, with the solid blue line showing the example ROI and the dashed line depicting the average spectral profile of the animal. e, Beta multiplier from the example ROI, depicting the deviation from the mean beta value for all ROIs from the same animal. f, Empirically measured spectral profiles from pure fluorophore samples, shown as beta-weighted contributors to ROI fingerprints. Scale bar: 100 µm (a, b), 10 µm (b inset and c).  

image, and the time-averaged GCaMP6s image from the behavioral session into a custom Python-based registration algorithm (Fig. 1f, 3b). Using a two-step process, first a coarse then a fine adjustment process, the code identified the optimal x-y shift and rotation by maximizing correlation coefficients. After applying these transformations to the GCaMP ROI masks (Fig. 3c), we extracted multiplexed spectral data for each ROI by averaging the fluorescence from all pixels within the mask to generate a unique spectral profile (Fig. 3d).  

# Spectral unmixing of in vivo ROI fingerprints  

Next, we unmixed the spectral data to identify which of the ten different fluorescent proteins were present in each ROI. To do so, we modeled the ROI spectral curve by using a linear unmixing algorithm, which determines the contribution of each fluorophore by best fitting a multiplier (beta) for each known spectral fingerprint (Fig. 1h, Fig. 3e-f) before summing the weighted spectral contributions. These reference spectra were derived empirically from HEK293T cells imaged using the same multiplexed spectral procedure.  

After fitting beta multipliers for each ROI, the spectral baseline was calculated by averaging the beta multipliers for each fluorophore over all ROIs within the FOV. This per-subject normalization step accounts for animal-to-animal variability in background signal and expression levels, which can significantly alter baseline spectral profiles (Supp. Fig. 3a–e). A fluorophore hit was assigned if the beta value for a given fluorophore exceeded the mean baseline beta by more than 1.5 standard deviations. Representative examples of fluorophore-classified ROIs from in vivo experiments are shown (Supp. Fig. 4a–r), illustrating the confocal image, co-registered GCaMP ROI, extracted spectral fingerprint, and corresponding beta values for each of the nine fluorophores used. This threshold was established by the simulation to balance sensitivity and specificity across subjects.  

# Assessment of spectral unmixing approach  

To determine the accuracy and robustness of our fluorophore identification algorithm component of Neuroplex, we validated the algorithm using a series of simulated datasets based on real spectral fingerprints. First, we expressed each fluorophore individually in HEK293T cells and collected spectral data from hundreds of cells per fluorophore. We then created test datasets by randomly combining spectral fingerprints in silico, allowing for any fluorophore composition. Additional perturbations were introduced, including increased GCaMP fluorescence (Supp. Fig. 5a–b), simulated overlap of multiple fluorophores within the same ROI (Supp. Fig. 5c–d), and white noise (Supp. Fig. 5e–f).  

In test datasets composed of equal numbers of single-fluorophore-expressing cells for all ten utilized fluorophores, the algorithm correctly estimated the beta contribution for each fluorophore to be approximately 10%, accurately reflecting the true distribution. Fluorophore identity was correctly assigned with nearly 100% accuracy across all fluorophores in the equal distribution dataset (Supp. Fig. 5g–h). Under realistic experimental conditions, however, the distribution of fluorophores is unlikely to be equal and will vary between animals (Supp Fig. 5i). To model this, we tested datasets in Phillips et al. Multicolor neuron identification within miniscope 7  

which one fluorophore made up an increasing proportion of the total population. When a single-fluorophore exceeded 30% of all ROIs, the algorithm's ability to identify that fluorophore dropped sharply (Supp. Fig. 5g–h), despite maintaining high accuracy for all other fluorophores. Upon closer inspection, assigned most ROIs belonging to the over-represented fluorophores as false-negatives (i.e., no match; Fig. 5h), as the fluorophore's spectral contribution deviated less from the spectral baseline.  

To recover these false negatives without compromising the specificity of the cells expressing other fluorophores, we implemented a dual-pass identification strategy. ROIs that failed to reach the threshold for any fluorophore in the first pass were re-evaluated in a second pass, in which the identification threshold was dynamically adjusted based on the measured distribution of beta values for each fluorophore from the spectral baseline (Supp. Fig. 5j). This dual-pass method successfully recovered over 90% of over-represented fluorophores, even when a single fluorophore accounted for 80% of the population, while retaining the unmodified high first-pass accuracy for the remaining fluorophores (Supp. Fig. 5g–h).  

We next tested the robustness of the unmixing algorithm under various imaging conditions. When increasing levels of GCaMP background were added to the spectra, identification accuracy declined gradually yet remained above 80% even when the GCaMP signal matched the fluorophore intensity (Supp. Fig. 5a–b). Importantly, this perturbation resulted primarily in false negatives without increasing the rate of false positives, thereby maintaining specificity at the cost of efficiency. We then modeled spatial overlap of cells expressing different fluorophores. As expected from our model,  

a)  

Fluorescent proteins

1. mTagBFP2

2. mTurquoise2

3. T-Sapphire

4. mVenus

5. mOrange2

6. mScarlet

7. FusionRed

8.mCyRFP1

9. mNeptune2.5

unidentified  

Brain regions d: dPAG

B: BLA
C: Cla
N: NAc
S: Str
L: LC
V: VTA
H: LHb
P: c-mPFC
y: LHyp  

c)  

# Fig. 4: Distribution of fluorophore-positive functionally defined ROIs  

a, Identified fluorophores for injection paradigm A. Viral injection paradigm A consisted of mTagBFP2 into the dPAG, mTurquoise2 into the BLA, T-Sapphire into the Cla, mVenus into the NAc, mOrange2 into the Str, mScarlet into the LC, FusionRed into the VTA, mCyRFP1 into the LHb, and mNeptune2.5 into the c-mPFC (left). Spatial distribution of ROIs and respective fluorophore matches overlaid on anatomical images from the same mouse (right). Distribution of identified fluorophores per mouse (inset). b, Identified fluorophores for injection paradigm B. Viral injection paradigm B consisted of mTagBFP2 into the c-mPFC, mTurquoise2 into the LHyp, T-Sapphire into the Str, mVenus into the VTA, mOrange2 into the LC, mScarlet into the dPAG, FusionRed into the BLA, mCyRFP1 into the NAc, and mNeptune2.5 into the Cla (left). Spatial distribution of ROIs and respective fluorophore assignments overlaid on the anatomical image from the same mouse (right). Distribution of identified fluorophores per mouse (inset). Color and letter codes for fluorophores and injection regions, respectively (far right). c, Percentage of identified ROIs with a fluorophore match. Animal n = 5; ROI n = 1,327. d, Percent of cells identified for each fluorophore. Letter insets on individual data points correspond to injected regions. N = 1,072. One-way ANOVA p = 0.0071. e, Percent of cells identified for each injected region. Number insets on individual data points correspond to injected fluorophore. N = 1,072. Mean ± SEM. One-way ANOVA, p = 0.2599. Scale bars: 100 µm.  

when the background contribution from a second fluorophore reached 50%, identification accuracy dropped to approximately 50%, again due mostly to false negatives (Supp. Fig. 5c–d). Finally, we introduced increasing levels of Gaussian white noise. Although accuracy declined slowly under this condition, we observed a modest increase in false positives, reaching roughly 10% at a signal-to-noise (SNR) ratio of 4 (overall accuracy approximately 80%; Supp. Fig. 5e–f).  

Finally, we estimated our fluorophore identification accuracy under realistic experimental conditions by simulating a dataset that mimicked key properties of an actual animal subject. For this dataset, we randomly sampled known fluorophore ROIs according to the measured fluorophore distribution across animals. We then added empirically measured background fluorescence and modeled GCaMP signal intensity at an average of 30% of the fluorophore brightness to reflect in vivo co-expression levels. To further challenge the algorithm, we introduced Gaussian white noise corresponding to a signal-to-noise ratio of 6 (Supp. Fig. 5k). Under these conditions, the dual-pass method correctly identified 87% of all ROIs, with a false positive rate below 5%. Identification accuracy varied modestly across fluorophores, reflecting differences in spectral proximity and intrinsic brightness (Supp. Fig. 5I).  

# Detection of two fluorophores within the same ROI  

Until now, we had only attempted to identify a single fluorophore alongside the GCaMP signal. Excitatory neurons, however, often project to multiple downstream targets via en passant boutons or bifurcating axons, opening the possibility that some neurons may express more than one retrograde label. To assess whether Neuroplex could resolve such dual-expressing neurons, we simulated a dataset in which ROIs expressed every possible pairwise combination of two different fluorophores, alongside single-labeled controls (Supp. Fig. 6a).  

Overall, the algorithm was able to correctly identify at least one fluorophore in 98% of dual-expressing ROIs, and both fluorophores in 44% of the cases. As expected, performance varied by spectral separation: fluorophore pairs with high spectral similarity (e.g., mScarlet + FusionRed) were harder to resolve, while spectrally distinct pairs (e.g., mScarlet + mVenus) were classified more reliably. When identification errors occurred, they were primarily false-negatives, with the algorithm failing to exceed the threshold for one of the fluorophores; false-positives made up less than 1% of all errors.  

Finally, we modeled the dual-expressing ROIs under realistic experimental conditions by adding empirically measured spectral background, GCaMP co-expression, and Gaussian white noise (Supp. Fig. 6b). Under these conditions, the algorithm correctly identified at least one fluorophore in 91% of ROIs, and both fluorophores in 25% of dual-labeled ROIs. Among ROIs containing only one fluorophore, the algorithm falsely assigned a second fluorophore in 17% of cases.  

Across all test ROIs, the overall false positive rate was 14%, with 5% occurring in the primary fluorophore assignment and 9% in the secondary fluorophore assignment.  

# Fluorophore distribution in behaviorally relevant ROIs  

To independently assess whether the combination of projection target and fluorophore identity influenced the classification outcomes, we designed two complementary injection paradigms across five mice (Fig. 4a–b). Each animal received the same set of nine AAVretro fluorophores, but in Group 2 the fluorophore-to-region pairings were rearranged to test detection robustness. Fluorophores with lower detection rates in Group 1 (Fig. 4a) were reassigned to brain regions with higher labeling efficiency in Group 2 (Fig. 4b).  

Across animal subjects, Neuroplex assigned a secondary fluorophore identity to 75% of behaviorally defined ROIs (Fig. 4c). ROIs were extracted from calcium imaging using CNMF, such that only neurons with activity during the behavioral session were included in the dataset. This approach inherently excludes silent but anatomically labeled cells from analysis, and thus the classification rate reflects the proportion of functionally active neurons with identifiable fluorophores. Each animal had between 7 and 9 of the injected fluorophores successfully detected, with 1,327 neurons identified as active during behavior and 1,156 of those being annotated with fluorophore identity (Fig. 4, Supp. Fig. 3). Detection rates varied between fluorophores, and these differences were not solely attributable to projection target. For example, mVenus and T-Sapphire had the highest detection frequencies. While T-Sapphire was consistently associated with regions exhibiting high expression across both groups, mVenus was preferentially detected compared to the other fluorophore co-injected into the same region. In contrast, FusionRed and mCyRFP1 were identified least frequently, despite being assigned to brain regions where their co-injected fluorophore was readily detected (Fig. 4d). This likely reflects lower brightness or expression of the fluorophores themselves and reflects predictions from in silico modeling (Supp. Fig. 5I).  

Projection-defined detection patterns also emerged. Neurons projecting to the claustrum (Cla), striatum (Str), and ventral tegmental area (VTA) were detected most frequently, consistent with known dense innervation from the prelimbic cortex (PL). Conversely, neurons projecting to the contralateral mPFC (c-mPFC), lateral habenula (IHb) and dorsal periaqueductal gray (dPAG) were detected less often. (Fig. 4e, Supp. Fig. 3).  

We next assessed how often functionally defined ROIs exceeded threshold for more than one fluorophore. Forty percent of ROIs had beta multipliers above threshold for a second fluorophore (Supp. Fig. 7a). These secondary hits were most common among fluorophores at the edges of the spectrum, including mTagBFP2, mTurquoise2, mCyRFP1, and mNeptune2.5 (Supp. Fig. 7b). Under high-background conditions, these fluorophores may be more permissive to co-occurring signals due to reduced spectral interference with other fluorophores. When grouped by brain region, dPAG, c-mPFC, and BLA-projecting neurons showed the highest rates of secondary fluorophore assignment (Supp. Fig. 7c). The distribution of double-labeled ROIs, however, did not align with known patterns of projection convergence from prior anatomical studies$^{24}$. This discrepancy likely reflects differential fluorophore expression and variable discriminability of fluorophore combinations. For studies where precise identification of multi-labeled neurons is critical, we recommend pairing spectrally distinct fluorophores for regions with known or expected projection overlap during experimental design. Representative in vivo examples of neurons with two spectrally distant fluorophores above threshold (ROI 28), two more similar fluorophores above threshold (ROI 98), and a neuron with one fluorophore above and a second just under the detection threshold (ROI 142) are shown in Supplementary Fig. 7d–f.  

# Neuronal cell types and behavior  

We applied our Neuroplex pipeline for a social memory assay. In this assay, animals were trained to recognize a familiar conspecific, followed by a test session where both the familiar and a novel mouse were introduced into the arena (Supp. Fig. 8a). We extracted spatial and temporal components of neuronal activity from miniscope videos using constrained non-negative matrix factorization for endoscopic data (CNMF-E) implemented through the Inscopix Data Processing Software. This approach identified spatial footprints and corresponding calcium traces for functionally active neurons  

during behavior. We then stratified these neurons by projection target and examined behaviorally selective activity across cell types.  

For example, NAc-projecting neurons were selective to social interactions with either familiar or novel conspecifics (Fig. 5a, c). Alternatively, LC-projecting neurons were selective for aggressive versus investigative social interactions (Fig. 5a). Furthermore, certain populations display greater selectivity for what behaviors they encode compared to others, as shown by looking at the number of distinct behaviors for which individual neurons statistically modulated their calcium  

a, Representative trace (left) and averaged ΔF/F traces (right) of calcium transients time-locked to behavioral annotations. Top traces depict a nucleus accumbens-projecting neuron with annotations denoting social interaction with either a familiar (black) or novel (grey) conspecific. Bottom traces denote a locus coeruleus-projecting neuron with annotations denoting aggressive (black) or investigative (grey) social interactions, regardless of conspecific target. Scale bars = 10 ΔF/F y-axis, 2 s (x-axis, left) and 5 ΔF/F (y-axis), 1 s (x-axis, right). N = 34 behavioral epochs for familiar, 38 epochs for novel, 32 epochs for aggressive, and 53 epochs for investigative interactions. Mean ± SEM. Two-sided t-test: familiar max response vs. novel, p = 0.0011; aggressive vs. investigative, p = 0.049. b, Distribution of the number of different behaviors for which each neuronal cell type statistically modifies its firing rate. c, Schematic of neuronal cell-types and the behavioral categories for which each cell encodes.  

activity. Indeed, lHyp-projecting neurons are either not selective or selective for only one specific type of behavior, whereas some Str-projecting neurons significantly modulate their activity for to up to six unique behaviors (Fig. 5b, c). While examples of selective neurons of each type could be found for most behaviors, encoding was enhanced in certain populations. Using this approach, our results indicate that Str-projecting neurons are preferentially modulated for aggressive behaviors, most notably with familiar conspecifics (Fig. 5c and Supp. Fig. 6b) as they statistically decreased their firing rate during the behavior (Supp. Fig. 8b).  

# Performance under reduced fluorophore complexity  

To evaluate the performance of Neuroplex under reduced-complexity, we applied the pipeline to two GCaMP transgenic animals injected with a subset of four fluorophores: mTagBFP2 (c-mPFC), mVenus (Str), mOrange2 (Cla), and mNeptune2.5 (VTA) (Fig. Supp. 9a). Detection frequency and identification accuracy were assessed using both experimental and modeled datasets.  

Despite differences in the number of functionally defined neurons between animals, Neuroplex successfully detected all four injected fluorophores in both subjects, with expected variability in labeling efficiency across regions (Supp. Fig. 9b-c). Across both animals, 57% of functionally defined neurons were assigned a fluorophore identity, and 15% of these were classified as co-expressing a second fluorophore- consistent with expectations based on known projection overlap in the mPFC (Supp. Fig. 9 d-f).  

To estimate classification accuracy, we generated model datasets that approximated experimental conditions by replicating observed fluorophore distributions, background fluorescence, GCaMP co-expression and added Gaussian white noise. When restricted to a single fluorophore identification, the pipeline achieved approximately 90% accuracy with fewer than 10% false positives (Supp. Fig. 9g). When applied to simulated dual-labeled cells, Neuroplex correctly identified at least one fluorophore in 92% of cases and successfully resolved both fluorophores in 20% (Supp. Fig. 9h). Finally, we assessed how specific fluorophore pairings influenced co-identification performance. As expected, spectrally distinct fluorophores (such as mTagBFP2 and mNeptune2.5) were more reliably separated than spectrally overlapping pairs (such as mVenus and mOrange2) (Supp. Fig. 9i). These results validate Neuroplex's ability to generalize to reduced-complexity labeling strategies and emphasize the importance of strategic fluorophore selection based on co-expression likelihood and system-specific constraints.  

# Discussion  

Our study establishes a framework for high-dimensional cell-type-resolved functional imaging in freely behaving animals. By integrating miniscope recordings with multiplexed confocal spectral fingerprinting through the same GRIN lens, Neuroplex addresses key spectral limitations of head-mounted microscopy—enabling simultaneous distinction of nine neuronal populations alongside GCaMP activity. We demonstrate that Neuroplex allows for the identification of up to 9 fluorophore-labeled neuronal subtypes during in vivo calcium imaging through a GRIN lens. By combining functional recordings from a head-mounted miniscope with multiplexed spectral confocal imaging in the same animal, Neuroplex supports the assignment of fluorophore identity to functionally defined neuronal locations without relying on post-fixation tissue processing. This approach allows for simultaneous monitoring of diverse neuronal populations within a single subject, facilitating direct comparisons of activity patterns across cell types during behavior. In this study, we demonstrate classification of up to ten spectrally distinct fluorophores—including the 9 injected fluorophores and genetically-expressed GCaMP—in medial prefrontal cortex (mPFC) neurons defined by their downstream projection targets.  

While several existing methods identify neuronal subtypes following behavior, most rely on post hoc approaches such as immunohistochemistry or in situ labeling. These techniques require co-registration between live imaging and fixed tissue, a process complicated by chromatic aberrations from the GRIN lens and nonuniform tissue distortion during fixation and mounting. In practice, this often necessitates labor-intensive manual, nonlinear alignment. Neuroplex circumvents these limitations by acquiring both structural and functional data in vivo through the same GRIN lens, using  

matched fields of view and consistent optical paths. Image registration is performed using a Python-based workflow that automatically computes optimal linear transformations—guided by anatomical landmarks such as blood vessel patterns—ensuring reproducible alignment across animals and sessions. This approach requires only x/y shifts, rotation, and scaling, avoiding the need for nonlinear warping or manual intervention if care is taken to ensure parallel alignment between the GRIN surface and the detector. By eliminating post-fixation alignment, Neuroplex not only improves accuracy but also preserves animals for longitudinal studies, greatly enhancing experimental efficiency and reproducibility.  

A key advantage of Neuroplex is its compatibility with longitudinal designs. Because both functional and spectral imaging are performed in vivo through the same implanted GRIN lens, the same neuronal populations can be assessed across multiple time points within a single subject. This enables chronic tracking of fluorophore-defined subtypes as animals undergo learning, disease progression, or therapeutic intervention—offering a level of circuit resolution not feasible with post fixation methods. Additionally, spectral imaging can be performed prior to behavioral testing, allowing researchers to verify expression patterns and exclude poorly labeled subjects before committing to lengthy experiments. One important consideration is that multiplexed spectral imaging requires animals to be anesthetized, particularly when using fluorophores that spectrally overlap with GCaMP. Without silencing, large fluorescence transients from GCaMP can obscure dimmer fluorophore signals and reduce unmixing fidelity, leading to an increased false positive rate.  

Neuroplex is compatible with a variety of GRIN lens types and sizes, as well as with cranial windows. In this study, we evaluated several commonly used GRIN lenses and found that all were usable when proper chromatic corrections were applied. We selected the widely used 1×4 mm silver-doped GRIN lenses for our experiments, given their high numerical aperture (NA) and reliable performance in vivo. While lithium-doped lenses exhibited improved spectral characteristics, including reduced chromatic aberrations, reduced wavelength-dependent transmission, and lower background fluorescence, they also had significantly lower NA, resulting in substantially dimmer signals. This tradeoff rendered them impractical for many in vivo settings, where fluorophore brightness and signal-to-noise ratio are often less than ideal. In practice, any GRIN lens that provides adequate optical throughput and allows for full-spectrum z-stacks within the microscope's travel range can be used with Neuroplex, provided appropriate chromatic correction is applied.  

Neuroplex is scalable and broadly adaptable to diverse circuit-mapping goals. While this study focused on projection-defined pyramidal neurons in the mPFC using nine fluorophores plus GCaMP, the pipeline is equally compatible with genetically defined subtypes, developmental stages, or activity-tagged populations. Many experiments may require fewer labels, allowing users to prioritize only the brightest and most spectrally distinct fluorophores for enhanced separation fidelity. This can be especially advantageous in experiments where multiple labels are expected within a single neuron—such as studies of projection convergence or co-expression—where careful fluorophore selection can substantially improve identification accuracy. In two experimental animals expressing four fluorophores plus GCaMP, Neuroplex maintained high classification accuracy (\~90%) with low false positive rates and successfully identified dual-labeled neurons, demonstrating strong performance even under reduced spectral complexity. These results highlight the flexibility of the approach for both high- and low-complexity applications, across a variety of biological contexts and imaging constraints. Fluorophore selection can also be tailored to the excitation and emission capabilities of individual imaging systems. For instance, microscopes equipped with near-infrared detection may benefit from using fluorophores such as iFP2.0, which emit more efficiently in the 700–740 nm range compared to mNeptune2.5.  

The ability to co-register multiple fluorophore markers with functional calcium data will substantially enhance not only the throughput of neural activity studies, but also the statistical power of circuit-level analyses. By enabling multiplexed classification of neuronal subtypes within the same animal, Neuroplex facilitates more precise, efficient, and scalable interrogation of brain function. Furthermore, the pipeline's compatibility with longitudinal studies and avoidance of post-hoc tissue processing make it particularly valuable for investigating dynamic processes like learning, adaptation, or disease progression at circuit-level resolution. Overall, our integrative approach opens new avenues for dissecting the cellular architecture of behavior across time, space, and experimental contexts.  
| Fluorophore | Laser | Interfering Fluorophore | Reason for ranking |
| --- | --- | --- | --- |
| mOrange2 | 514 nm | NA | Bright, distinct |
| mTagBFP2 | 405 nm | NA | Several distinct spectral bins |
| mVenus | 514 nm | NA | Bright |
| mTurquoise2 | 405 nm*Would benefit from nm | mTagBFP2 & T-Sapphire | Neutral |
| T-Sapphire | 405 nm | NA | Neutral |
| FusionRed | 594 nm | mScarlet | Underperformer, slightly dim |
| mScarlet | 561 nm | mOrange2 & mNeptune2.5 | Cross talk |
| mNeptune2.5 | 639 nm | mScarlet | Dim, cross talk |
| mCyRFP2 | 488 nm | NA | Very dim |  

# Materials and Methods  

# Code and Data availability  

Code and data utilized in Neuroplex multispectral detection: GitHub - Neurocipher/PythonPipeline: Python pipeline for Neurocipher  

Code and data utilized in calcium / behavior correlation: https://github.com/MetaCell/Zeiss-Data-Science  

A detailed tutorial on these processes can be found via this repository:  

https://zeiss.tourial.com/dc/MultiColorInVivoImaging  

# Ethics & Inclusion statement  

We have carefully considered research contributions and authorship criteria during collaborations so as to promote greater equality in research.  

# Mice  

All experimental procedures were approved by the Max Planck Florida Institute for Neuroscience Institutional Animal Care and Use Committee and were performed in accordance with guidelines from the US NIH. Mice were group housed in 12h light-dark cycle with food and water ad libitum. The mice used in this study resulted from crossing heterozygous or homozygous B6.DBAGt(tetO-GCaMP6s)2Niell/J mice (JAX 024742) to heterozygous B6.Cg-Tg(Camk2a-tTA)1Mmay/DboJ (JAX 007004).  

# Fluorophore selection  

# Initial selection  

Fluorophores were identified based on spectral profiles published on fpbase.com and chosen for further study based on the specific spectral fingerprints we hypothesized would be distinguishable by multiplexed spectral imaging. These fluorophores included: mTagBFP2, mTurquoise2, T-Sapphire, mVenus, mPapaya, mOrange2, mScarlet, FusionRed, mCyRFP1, and mNeptune2.5.  

# In vitro expression  

Vectors containing the identified fluorophores obtained from Addgene (Addgene.com) were cloned into plasmids under the CMV promoter. HEK293T cells (GE Dharmacon, Fisher Scientific) were cultured in DMEM supplemented with 10% FBS at 37°C in 5% CO2 and transfected with plasmids using Lipofectamine 1000 (Invitrogen). Imaging was performed 24-  

h following transfection. HEK293T cells were used as an expression platform only and were not rigorously tested for potential contamination from other cell lines.

# Viral construction and evaluation  

Plasmids encoding mTagBFP2, mTurquoise2, T-Sapphire, mPapaya, mOrange2, mScarlet, FusionRed, mCyRFP1, and mNeptune2.5 were introduced into a hSyn-mVenus vector, replacing the mVenus sequence. The resulting vectors were used to create AAV$_\text{retros}$ (UNC vector core). Each virus was tested by injecting into the right medial prefrontal cortex of a wildtype mouse. After waiting three weeks for expression, mice were euthanized and 100 µm coronal sections containing the medial prefrontal cortex were prepared. Fluorophores were evaluated for expression levels, brightness, photostability, and neuronal death. mPapaya was excluded from further experiments due to excessive neuronal death after viral expression.  

# Surgery  

Mice were anesthetized with 4% isoflurane vapor in 100% oxygen gas and maintained with 1–2.5% isoflurane vapor in 100% oxygen gas mixtures. Mice were aligned in a stereotactic frame (Kopf Instruments), and their body temperature was measured with a rectal probe and maintained at 37°C with a heating pad. Warmed sterile saline was injected subcutaneously at a rate of 0.1 mL/hour to maintain hydration. Additionally, mice received subcutaneous injections of carprofen (5mg/kg) and dexamethasone (0.2 mg/kg) to reduce inflammation. Mice were monitored post-surgically and returned to their respective homecage once ambulation returned to normal.  

# For viral evaluation  

A midline incision was made down the scalp, and a dental drill was used to perform a small craniotomy over the right medial prefrontal cortex. A 2.5 μL syringe (Hamilton Company) was used to inject 250 nL of virus at a rate of 0.25 μL/min using a microsyringe pump (UMP3 UltraMicroPump, Micro4; World Precision Instruments), for coordinates see Table 1. The needle was slowly extracted from the injection site over 10 min. The scalp was closed using surgical glue.  

# For multi-color GRIN experiments  

A midline incision was made down the scalp, the scalp lightly scored to improve adherence of dental cement, and a dental drill used to perform a small craniotomy over the target area. A 2.5 μL syringe (Hamilton Company) was used to inject viruses at a rate of 0.25 μL/min using a microsyringe pump (UMP3 UltraMicroPump, Micro4; World Precision Instruments), see Table 2 for coordinates and volumes. The needle was slowly extracted from the injection site over 10 min. This injection procedure was repeated for each of the 9 fluorophore viruses, each injected into a unique region. Once all viral injections were complete, a 1.2 mm diameter craniotomy was performed over the mPFC and the dura carefully removed. A custom-made metal probe measuring 1mm in diameter was lowered at a rate of 100 μm/min to reach 300 μm above the desired imaging plane. The probe was left in place for 15 minutes before being retracted slowly over 10 minutes. Immediately afterward, a 1 x 4 mm silver-doped GRIN lens with integrated baseplate ("1x4 mm regular", Inscopix) and head-bar was lowered into place at a rate of 100 μm/min. Any area between the craniotomy and GRIN lens was sealed with silicone (KWIK-SIL, World Precision Instruments) after which the implant was secured in place using dental cement. The skin was sutured around the implant (DemeTech). Mice were given at least 5 weeks to recover from surgery before use in experiments.  
| Acronym | Region | Angle | Medial/Lateral | Anterior/Posterior | Dorsal/Ventral | Volume |
| --- | --- | --- | --- | --- | --- | --- |
| dPAG | Dorsal periaqueductal gray | 26° | 1.18 | -4.2 | 2.36 | 300 nL |  

| BLA | Basolateral amygdala | 0° | 2.95 | -1.6 | 3.8 3.3 | 250 nL 250 nL |
| --- | --- | --- | --- | --- | --- | --- |
| NAc | Nucleus accumbens | 0° | 1.0 | 1.3 | 4.0 3.5 | 250 nL 250 nL |
| Str | Striatum | 0° | 1.25 | 1.3 | 2.5 2.0 | 250 nL 250 nL |
| Cla | Clastrum | 0° | 2.2 | -1.7 | 2.6 | 300 nL |
| LHyp | Lateral hypothalamus | 0° | 1.1 | -1.3 | 5.3 4.75 | 250 nL 250 nL |
| LHb | Lateral habenula | 0° | 0.5 | -1.6 | 2.75 2.5 | 250 nL 250 nL |
| LC | Locus coeruleus | 0° | 0.8 | -5.3 | 3.0 3.0 | 250 nL 250 nL |
| VTA | Ventral tegmental area | 0° | 0.45 | -3.0 | 4.25 3.45 | 250 nL 250 nL |
| c-mPFC | Contra lateral medial prefrontal cortex | 0° | 0.4 | 1.5 | 1.45 | 300 nL |  

# Mouse perfusion  

Mice were anesthetized with an intraperitoneal (i.p.) injection of a ketamine (100 mg/kg) and xylazine (50 mg/kg) and transcardially perfused with ice-cold 1X phosphate-buffered saline (PBS), followed by ice-cold 4% paraformaldehyde (PFA) in 1X PBS. The brain was dissected and post-fixed in 4% PFA overnight at 4°C. Brains were sectioned at 100 µm with a vibratome (VT1200, Leica) and mounted with Vectashield mounting media (Vector Laboratories).  

# Behavioral experiments  

# Testing  

One week prior to testing, the back of sentinel mice was dyed with blonde hair dye (Born Blond Maxi, Clairol) with differing patterns for tracking by computer vision. The behavioral paradigm consisted of 5 days, with one behavior session per day. At the beginning of each behavioral session, test mice were acclimated to the behavior chamber alone for 10 minutes. On days 1–4, the same sentinel mouse (initially novel) was added to the behavior chamber following the acclimation period and allowed to freely interact for 10 minutes. On day 5, the trained sentinel mouse and a second novel sentinel mouse were added to the behavior chamber following the acclimation period and allowed to freely interact for 10 minutes. The test box was cleaned and filled with new bedding between each session. Each sentinel mouse interacted with a maximum of 3 test mice per 5-day paradigm. Custom-written MATLAB (Mathworks) code was used to record behavioral videos at 20 Hz.  

# Analysis  

After all mice had been tested, sentinel mice were individually videotaped for 10 min for generating training data. Individual and test videos were fed to the Motr program (https://github.com/motr/motr$^{29}$) to create tracks that were sent to JAABA (https://github.com/kristinbranson/JAABA$^{30}$) for unbiased computer identification of behaviors. JAABA classifiers were trained on pilot data sets.  

# Imaging  

In vivo calcium imaging using the miniscope  

On each day of behavioral experimentation, the miniscope was mounted on the implanted GRIN lens and baseplate immediately prior to the mouse being placed in the behavioral chamber. Custom-written MATLAB code triggered simultaneous video acquisition and nVoke2 (Inscopix) calcium recording, both recording at 20 Hz sampling rate. Parameters of the miniscope, such as the LED power, the gain, and the electronic focus, were adjusted on a mouse-to-mouse basis but otherwise kept consistent for the 5 sequential days of behavioral testing.  

Acquired calcium transients, concatenated per recording day with each recording day consisting of the acclimation and behavior time, were processed in the Inscopix Data Processing Software (IDPS, Inscopix). First, the traces were spatially down sampled by a factor of 2, then preprocessed, spatially filtered, and motion corrected. Individual cells were identified using a constrained non-negative matrix factorization algorithm (CNMF). Traces with abnormal physiological calcium transients (i.e., transients lasting over one minute) were excluded. For co-registration purposes, the motion-corrected video was temporally averaged into an image depicting anatomical landmarks. ROIs generated by the CNMF and their corresponding calcium traces were also exported.  

# Multicolor volumetric confocal imaging  

The post-behavioral confocal imaging was performed using a LSM 980 confocal microscope on an Examiner Z.1 upright stage. The mouse treadmill was inserted directly onto the xy-mechanical stage, bypassing the z-piezo stage. Images were taken using a 10x, NA 0.4 objective lens (C Epiplan-APOCHromat, 422642-9900, Zeiss), and utilizing both multialkali PMTs (Ch1 and Ch2) and the GaAsP detector (ChS), spanning a wavelength range from nm – 750 nm resulting in 34 spectral bins. All six excitation lasers (405, 488, 514, 561, 594, and nm) were used during volumetric spectral imaging. The pinhole was set to 'optimal' when imaging without a GRIN lens, and to 350 µm when imaging through a GRIN lens. Detector gain voltages and laser powers were set for each animal to prevent detector saturation for the brightest spectral channels and kept constant for each multiplexed spectral image within each animal. MultiBeam splitters (MBS) were set to MBS 405, MBS 488/561/639, or MBS 455/514/594 depending on the excitation laser used. No dichroic secondary beam splitter was used. Imaging parameters included a zoom setting of 1.0, no averaging, and a scan speed setting of 5.  

# GRIN lens optical transmission determination  

A custom-built GRIN lens micromanipulator was used to suspend the GRIN lens above a platform. Using a photodiode power sensor (S121C, Thorlabs) and power meter (PM100D, Thorlabs), we recorded the laser power of the LSM 980 excitation lasers (ZEISS) out of the objective lens, first directly and second after focusing through a GRIN lens. The difference in these values was used to calculate the wavelength-dependent transmission through the GRIN lens. The data was fitted to a second-order polynomial, and subsequently used to pre-adjust the intensity of each excitation laser to ensure wavelength-independent laser power on the sample.  

# Measuring GRIN-induced chromatic aberrations  

Using a custom-built GRIN lens micromanipulator (MPFI), the GRIN lens was centered and suspended over the fluorescent "field of rings" pattern on an Argo-LM v2.0 slide (ArgoLight). The pattern was imaged through the GRIN lens using a 10x objective lens (0.4 NA, C Epiplan-APOCHromat, ZEISS). The upper and lower z-stack limits were set using the nm and nm lasers, respectively, after which the entire z-stack was imaged at 5 µm intervals for each excitation laser (405, 488, 514, 561, 594, and nm). To determine the chromatic shifts along the optical axis, the z-plane with the brightest intensity for the center crosshair pattern was determined to be the focal plane for each laser wavelength, and the wavelength – z focal plane relationship was fit to these values using a second order polynomial. There were negligible chromatic aberrations in the optical plane (i.e., the xy dimensions).  

# Multiplexed spectral imaging technique  

To enable discrimination of all 10 fluorophores, we utilized a "multiplexed spectral imaging" technique. This involved sequentially imaging the same field of view using six different excitation lasers (405, 488, 514, 561, 594, and nm)  

while detecting the emission in spectral mode using both multialkali (32 array detector) and the GaAsP PMTs (2 detectors), together collecting the fluorescence emissions in 34 separate spectral bins. The bin widths were approximately 10 nm between nm to 695 nm (longer wavelengths have larger emission bins) using the GaAsP detector, with the multialkali PMTs spanning the wavelengths from nm – 400nm and 695nm – 750 nm, respectively. This resulted in a spectral fingerprint consisting of measurements per ROI (6 excitation lasers x 34 emission bins).  

# Spectral fingerprint generation  

After allowing 24 hours for expression, wells containing fluorophore-expressing HEK293T cells were rinsed and filled with imaging buffer. Wells were imaged on a ZEISS LSM 980 at 10x magnification using the multiplexed spectral technique. Laser powers were determined per sample but kept the same for each excitation laser. The pinhole was set to optimal and one focal plane and field of view were taken per sample.  

For each fluorophore, the 6 corresponding images were imported into Fiji (ImageJ). The image with the highest intensity was selected and then thresholded to isolate cells from background. The Analyze Particles module was used to determine ROIs. The 204-point spectral fingerprint of each fluorophore was determined by averaging the multiplexed spectral traces from all ROIs from a single sample.  

# In vivo multiplexed spectral imaging  

Mice were anesthetized with an i.p. injection of ketamine (100 mg/kg) and xylazine (10 mg/kg) to reduce high-intensity fluctuations of GCaMP6s transients, though low-intensity slow-wave fluctuations remained. The GRIN lens was thoroughly cleaned with isopropanol and water and the mouse was head-fixed on a custom treadmill (MPFI). The temperature was monitored and maintained with hand warmers (Hothands). Images were obtained using the spectral imaging mode on a ZEISS LSM 980 confocal microscope running the ZEN Blue software. Using 10x magnification, the objective pinhole was set to 350 µm to minimize z-dimensional chromatic aberration of emission introduced by the GRIN lens. For each mouse, z-range was set by utilizing the nm and nm excitation lasers to account for the remaining z-dimensional chromatic aberration and sampled at 5 µm intervals with the typical range being 48 slices or 240 µm. The maximum intensity of all fluorophores was found, and the laser power of the corresponding laser was set to avoid oversaturation. Using this laser value, the remaining laser values were adjusted to account for differing wavelength-dependent transmission of the GRIN lens except for the nm laser. Because mNeptune2.5 is the only fluorophore to emit following 639 nm excitation, and due to the reduced efficiency of its excitation at this wavelength, we consistently set the nm laser to 40% power across all mice. Beginning with the nm laser and progressing sequentially, we recorded the entire z-stack in the spectral imaging mode for each laser. If high-intensity fluctuations for GCaMP were observed, the mouse was recorded on a subsequent day to eliminate these as much as possible.  

In post-processing, spectral z-stacks from each laser were cropped in the z-dimension to remove any high-noise z-planes, cropping was applied consistently to all images from the same mouse. Background subtraction was performed to reduce non-somatic signals. A summed Z-projection for each excitation laser stack was used for the spectral analysis.  

# Co-registration of functionally defined neurons  

# nVoke to Zeiss registration  

Prior to image registration, a series of preprocessing steps were applied to denoise and enhance salient structural features. A Gaussian filter with a small sigma value was first applied to reduce noise, with a sigma of 1 pixel applied to calcium imaging data and 2 pixels applied to confocal imaging data. To estimate and remove background, a Gaussian blur with a large sigma was applied to the image, and the resulting blurred background was subtracted from the original. The sigma used for this background estimation was 50 pixels for calcium imaging and pixels for confocal imaging. Following background subtraction, a morphological black-hat operation was applied to extract dark, vessel-like features. This operation highlights structures that are smaller than a specified window size and is particularly effective at isolating blood vessels, which serve as reliable features for subsequent registration. A window size of 11 pixels was used for  

calcium imaging and 21 pixels for confocal imaging. Image registration was then formulated as an optimization problem designed to maximize correlation between images by adjusting four transformation parameters: x-translation, y-translation, rotation, and global scaling (i.e., a similarity transformation). The registration process was conducted in two stages. First, a coarse, exhaustive grid search explored a defined parameter space, with translation ranges set to ±60 pixels, rotation to ±15 °, and scaling to a range of 1.8 to 2.0. All parameters were sampled at discrete intervals (5 pixels for translations, 5 degrees for rotation, and 0.05 for scaling). The best candidate from this global search was then refined using gradient descent to obtain a locally optimal solution. The learning rate for this fine-tuning stage was set to 0.5.  

# Fluorophore identification  

# Modeling of experimental variables to assess accuracy of algorithms  

To assess how the fluorophore identification algorithm was performing, we utilized wells containing single fluorophore expressing HEK293T cells imaged using the multispectral approach. We created a library of individual ROIs from each well and randomly sampled these ROIs to create simulated datasets. Each simulation evenly distributed the desired manipulation to ROIs of each fluorophore and was conducted in replicates of 100. Fluorophore matches assigned by the algorithm were compared to the known ROI identity, allowing us to measure the accuracy by determining the percentage of correct matches, false negatives (no match), and false positives (incorrect match). We tested the algorithm robustness under several simulated perturbations, including added background signal, unequal fluorophore distributions, and synthetic ROIs formed by combining spectral from different fluorophores. GCaMP background was modeled by adding scaled GCaMP spectra at varying intensities relative to the average fluorophore brightness, spectral background consisted of adding a uniform mixture of fluorophore spectra at varying proportions relative to the average fluorophore intensity, , and Gaussian white noise was added at varying signal-to-noise ratios. Initially, we modeled each condition separately to determine which perturbations created what types of errors. Finally, we simulated an experimental condition by matching conditions within an animal subject as closely as possible. To this end, we recreated the fluorophore population of the dataset to reflect the actual measured hits, added GCaMP background at a level of 30% (the measured peak of GCaMP signal within ROIs), spectral background obtained by averaging all pixels within the GRIN lens FOV at a rate of 30% (measured by comparing the peak intensity of background to the average peak intensity of fluorophore-containing ROIs), and Gaussian white noise at a signal-to-noise ratio of six as an estimate.  

To evaluate performance on dual-labeled ROIs, we generated datasets containing all pairwise combinations of two fluorophores. To this end, we randomly selected ROIs from HEK293T wells containing single fluorophores and created new ROIs by summing the spectra from two ROIs with each of the respective fluorophores. The resulting dataset contained equal numbers of ROIs expressing all possible dual-fluorophore pairs and ROIs expressing only a single fluorophore.  

# Single Pass algorithm  

The multiplexed spectral data from the functionally defined ROIs was evaluated to fit to the pure sample fingerprints using a linear regression. Beta values for each fluorophore were generated for ROIs using both the raw emission values ($\beta_{\text{raw}}$) and those normalized to maximum intensity ($\beta_{\text{norm}}$). For each mouse, a cutoff was determined as 1.5 standard deviations above mean for each fluorophore's beta values. It was necessary to do this on a mouse-by-mouse basis because background varied based on the number and ratio of neurons expressing fluorophores (Supp. Fig. 3). ROIs with $\beta_{\text{raw}}$ or $\beta_{\text{norm}}$ above these thresholds were considered positive for a fluorophore. If multiple fluorophores exceeded the threshold for an ROI, the fluorophore with the largest z-scored beta value was assigned.. Both $\beta_{\text{raw}}$ and $\beta_{\text{norm}}$ were used, as they exhibit complimentary biases for bright and dim ROIs, respectively. Cutoffs and procedures were evaluated using multiplexed spectral bins where only one fluorophore could be emitting.  

# Dual Pass algorithm  

To recover fluorophore assignments missed in by the first pass due to over-represented fluorophores, we implemented a second identification step for any ROIs which were not assigned a hit by the first pass. This second pass utilized the Phillips et al. Multicolor neuron identification within miniscope 19  

same linear regression and beta multiplier calculation as the first pass but adjusted these values to correct for uneven fluorophore distribution. To apply the correction, we first modeled a dataset with a perfect distribution of HEK293T cell ROIs from single fluorophore wells. We obtained the average beta multipliers for all ROIs and assessed the beta contribution of each fluorophore as a percent of total. These values were utilized as a theoretical equal distribution. We then determined the beta contribution for each fluorophore in our experimental conditions and adjusted the cutoff threshold for assignments based on the deviation from the theoretical value, resulting in proportionally lowered thresholds for over-represented fluorophores.  

# Linking neural activity to behavior and delineating by cell-type  

For each animal and session, we defined a behavior set appropriate to the session type. For example, single-animal behaviors (moving, being still) were defined for acclimation sessions, while social interaction behaviors (sniffing, aggression) were defined for training and testing sessions, with different conspecific targets (novel vs. familiar) in the testing session.  

Then, for each cell and each behavior in each session, a t-test was carried out comparing mean activity in the 2.5-second window before versus after behavioral onset with Bonferroni correction accounting for multiple comparisons across different behaviors. If the mean activity was significantly different before and after a given behavior, the cell was classified as behaviorally selective. The sunburst diagram visualizes the number of cells that were responsive to different behaviors, separated by different brain regions. The sankey diagram visualizes the relative proportion of cells that were responsive to different behaviors separated by different brain regions.  

# Acknowledgments  

We would like to acknowledge David Kloetzer for lab management, Yuki Hayano and Irena Suponitsky-Kroyter for technical assistance, and the MPFI ARC, including Elizabeth Garcia, and Amanda Coldwell for animal care and maintenance. This work was supported by National Institutes of Health Grants R35-NS-116804 (RY), R01-MH-080047 (RY), and F32MH120872 (M.L.P.).  

# References  

1. Resendez, S. L. & Stuber, G. D. In vivo Calcium Imaging to Illuminate Neurocircuit Activity Dynamics Underlying Naturalistic Behavior. Neurropsychopharmacology 40, 238–239 (2015).

2. Ghosh, K. K. et al. Miniaturized integration of a fluorescence microscope. Nat. Methods 8, 871–878 (2011).

3. Barbera, G., Liang, B., Zhang, L., Li, Y. & Lin, D.-T. A wireless miniScope for deep brain imaging in freely moving mice. J. Neurosci. Methods 323, 56–60 (2019).

4. Guo, C. et al. Miniscope-LFOV: A large field of view, single cell resolution, miniature microscope for wired and wire-free imaging of neural dynamics in freely behaving animals. 2021.11.21.469394 Preprint at https://doi.org/10.1101/2021.11.21.469394 (2022).

5. Scott, B. B. et al. Imaging Cortical Dynamics in GCaMP Transgenic Rats with a Head-Mounted Widefield Macroscope. Neuron 100, 1045-1058.e5 (2018).

6. Helmchen, F., Fee, M. S., Tank, D. W. & Denk, W. A Miniature Head-Mounted Two-Photon Microscope: High-Resolution Brain Imaging in Freely Moving Animals. Neuron 31, 903–912 (2001).

7. Zong, W. et al. Fast high-resolution miniature two-photon microscopy for brain imaging in freely behaving mice. Nat Methods 14, 713–719 (2017).

8. Zong, W. et al. Large-scale two-photon calcium imaging in freely moving mice. Cell 185, 1240-1256.e30 (2022).

9. Skocek, O. et al. High-speed volumetric imaging of neuronal activity in freely moving rodents. Nat. Methods 15, 429-432 (2018).

10. Stamatakis, A. M. et al. Miniature microscopes for manipulating and recording in vivo brain activity. Microscopy 70, 399–414 (2021).

11. Aharoni, D. & Hoogland, T. M. Circuit Investigations With Open-Source Miniaturized Microscopes: Past, Present and Future. Front. Cell. Neurosci. 13, (2019).

12. Kohl, J. et al. Functional circuit architecture underlying parental behaviour. Nature 556, 326–331 (2018).

13. Kingsbury, L. et al. Cortical Representations of Conspecific Sex Shape Social Behavior. Neuron 107, 941-953.e7 (2020).

14. Langer, D. & Helmchen, F. Post hoc immunostaining of GABAergic neuronal subtypes following in vivo two-photon calcium imaging in mouse neocortex. Pflugers Arch. 463, 339–354 (2012).

15. Kahan, A. et al. Light-guided sectioning for precise in situ localization and tissue interface analysis for brain-implanted optical fibers and GRIN lenses. Cell Rep. 36, 109744 (2021).

16. Xu, S. et al. Behavioral state coding by molecularly defined paraventricular hypothalamic cell type ensembles. Science 370, eabb2494 (2020).

17. Khan, A. G. et al. Distinct learning-induced changes in stimulus selectivity and interactions of GABAergic interneuron classes in visual cortex. Nat. Neurosci. 21, 851–859 (2018).

18. Lambert, T. J. FPbase: a community-editable fluorescent protein database. Nat. Methods 16, 277–278 (2019).

19. Lee, W. M. & Yun, S. H. Adaptive aberration correction of GRIN lenses for confocal endomicroscopy. Opt. Lett. 36, 4608–4610 (2011).

20. Royon, A. & Converset, N. Quality Control of Fluorescence Imaging Systems: A new tool for performance assessment and monitoring. Opt. Photonik 12, 22–25 (2017).

21. Wekselblatt, J. B., Flister, E. D., Piscopo, D. M. & Niell, C. M. Large-scale imaging of cortical dynamics during sensory perception and behavior. J. Neurophysiol. 115, 2852–2866 (2016).

22. Tervo, D. G. R. et al. A Designer AAV Variant Permits Efficient Retrograde Access to Projection Neurons. Neuron 92, 372–382 (2016).

23. Zhou, P. et al. Efficient and accurate extraction of in vivo calcium signals from microendoscopic video data. eLife 7, e28728.

24. Gongwer, M. W. et al. Mapping whole-brain projections of anatomically-defined prefrontal neurons using combined 3D convolutional networks. 2022.03.31.486619 Preprint at https://doi.org/10.1101/2022.03.31.486619 (2022).

Phillips et al. Multicolor neuron identification within miniscope 21  

25. Meier, A. F., Fraefel, C. & Seyffert, M. The Interplay between Adeno-Associated Virus and Its Helper Viruses. Virus 12, (2020).

26. Dembrow, N. & Johnston, D. Subcircuit-specific neuromodulation in the prefrontal cortex. Front. Neural Circuits 8, 54 (2014).

27. Remedios, R. et al. Social behaviour shapes hypothalamic neural ensemble representations of conspecific sex. Nature 550, 388–392 (2017).

28. Kennedy, A. et al. Stimulus-specific hypothalamic encoding of a persistent defensive state. Nature 586, 730–734 (2020).

29. Ohayon, S. et al. Automated multi-day tracking of marked mice for the analysis of social behavior. Journal of Neuroscience Methods 219, 10–19 (2013).

30. Kabra, M. et al. JAABA: interactive machine learning for automatic annotation of animal behavior. Nature Methods 10, 64–67 (2013).  

# Supplementary Figures  

a)  

c)  

b)  

# Supp. Fig. 1: Comparison of aberrations across GRIN lens types  

a, Shift in z-focal plane as a function of excitation laser wavelength in a 1x4 mm silver-doped GRIN lens. Second-order polynomial R$^{2}$ = 0.9970. b, Shift in z-focal plane as a function of excitation laser wavelength in 0.6x7 mm GRIN lenses doped with either silver or lithium. Second order polynomials: Silver R$^{2}$ = 0.9983, Lithium R$^{2}$ = 0.9928. c, Percent transmission through a 1x4 mm silver-doped GRIN lens as a function of excitation laser wavelength. Sixth-order polynomial R$^{2}$ = 0.9926. d, Percent transmission through 0.6x7 mm GRIN lenses doped with either silver or lithium as a function of excitation laser wavelength. Sixth-order polynomial: silver, R$^{2}$ = 0.9979; lithium, R$^{2}$ = 0.9549.  

Supp. Fig. 2: Multiplexed spectral fingerprints for single-fluorophore samples

a, Spectral fingerprints for each fluorescent proteins were obtained by measuring emission intensities using multiplexed spectral imaging in transfected HEK293T cells.  

| Supp. Fig. 3: Subject-to-subject variation |
| --- |
| a–e, Average spectral fingerprints computed across all ROIs for each experimental mouse. Mean ± 1.5 x SD. f–j, Spatial distribution of ROIs and their corresponding fluorophore assignments overlaid on anatomical images from the same mouse. Scale bars = 100 µm. |  

a)  

b)  

c)  

e)  

1. dPAG: mTagBFP2, n = 10
2. Cla: mTurquoise2, n = 7
3. BLA: T-Sapphire, n = 13
4. NAC: mVenus, n = 9
5. Str: mOrange2, n = 14
6. LC: mScarlet, n = 3
7. VTA: FusionRed, n = 4
8. LHb: mCyRFP1, n = 5
9. c-mPFC: mNeptune2.5, n = 7
All cells, n = 105  

1. dPAG: mTagBFP2, n = 20
2. Cla: mTurquoise2, n = 23
3. BLA: T-Sapphire, n = 39
4. NAC: mVenus, n = 17
5. Str: mOrange2, n = 102
6. LC: mScarlet, n = 68
7. VTA: FusionRed, n = 28
8. LHb: mCyRFP1, n = 23
9. c-mPFC: mNeptune2.5, n = 30
All cells, n = 440  

1. dPAG: mTagBFP2, n = 9
2. Cla: mTurquoise2, n = 8
3. BLA: T-Sapphire, n = 42
4. NAC: mVenus, n = 11
5. Str: mOrange2, n = 7
6. LC: mScarlet, n = 7
7. VTA: FusionRed, n = 6
8. LHb: mCyRFP1, n = 2
9. c-mPFC: mNeptune2.5, n = 12
All cells, n = 178  

1. c-mPFC: mTagBFP2, n = 17
2. Str: mTurquoise2, n = 13
3. LHyp: T-Sapphire, n = 27
4. VTA: mVenus, n = 68
5. LC: mOrange2, n = 7
6. dPAG: mScarlet, n = 7
7. BLA: FusionRed, n = 0
8. NAC: mCyRFP1, n = 1
9. Cla: mNeptune2.5, n = 11
All cells, n = 213  

1. c-mPFC: mTagBFP2, n = 27
2. Str: mTurquoise2, n = 41
3. LHyp: T-Sapphire, n = 158
4. VTA: mVenus, n = 117
5. LC: mOrange2, n = 14
6. dPAG: mScarlet, n = 9
7. BLA: FusionRed, n = 3
8. NAC: mCyRFP1, n = 7
9. Cla: mNeptune2.5, n = 19
All cells, n = 395  

# Supp. Fig. 4: Representative examples of fluorophore-identified neurons using Neuroplex  

a-r, Each panel shows an ROI that exceeded threshold for a single fluorophore identity assignment. Left: Functional ROIs identified from miniscope recordings during behavior, co-registered and overlaid on corresponding in vivo confocal images. Center: Spectral fingerprint of the ROI (solid line), compared to the animal's average spectral background (dashed line). Excitation-emission bins are color-coded to excitation laser wavelength. Right inset: Beta multiplier values for all fluorophores from the same ROI, plotted as standard deviations above the animal-specific baseline. Scale bars = 10 µm.  

# Supp. Fig. 5: Modeling-based evaluation of fluorophore identification robustness under challenging conditions  

a–h, Simulated single-fluorophore datasets were used to assess Neuroplex classification performance under conditions designed to mimic common experimental noise sources. Each panel shows a schematic of the modeling setup (inset), average identification accuracy for single-pass (gray) and dual-pass (black) classification (top), and a breakdown of classification outcomes by fluorophore, including correct identifications, false-negatives, and false-positives (bottom). Each simulation contained 250 ROIs and was repeated 100 times. Modeled perturbations included: increasing GCaMP background within ROIs (a–b), added low-level spectral background from other fluorophores (c–d), decreased signal-to-noise ratio via Gaussian white noise (e–f), and over-representation of a single fluorophore in the population (g–h). i, Comparison of theoretical equal beta contributions (dashed line) to empirically observed beta weights across fluorophores in an example animal, showing deviations that inform the need for adjusted thresholds. j, Example ROI illustrating how second-pass analysis recovers fluorophore assignments missed during initial thresholding. k, Simulated experimental condition using known ROI spectra matched to the actual fluorophore distribution in animal YAS21272R. Background fluorescence, GCaMP co-expression, and white noise were added to approximate in vivo signal characteristics. l, Final classification breakdown for the modeled condition in k, using the dual-pass approach.  

a) GCaMP background  

b)  

c) Fluorophore background  

e) White noise  

f)  

g) Over-represented fluorophores i) Experimental Distributions  

j)  

k) Experimental Conditions  

1)  

# a) Modeling conditions where ROIs may express up to two fluorophores  

b) Modeling experimental conditions where ROIs may express up to two fluorophores  

# Supp. Fig. 6: Modeling dual fluorophore expressing ROIs  

a, Modeled conditions with ROIs containing either single or dual fluorophores: depiction (left), breakdown of dual-pass analysis performance per fluorophore pair (right). b, Modeled experimental conditions with ROIs containing either single or dual fluorophores with added experimental background, GCaMP background, and white noise: depiction (left), breakdown of dual-pass analysis performance (right). Bars reflect percent of ROIs where each fluorophore was correctly identified (colored), misidentified (white), or not identified (grey). ROIs n = 460, replicates n = 100.  

a)  

c)  

# Supp. Fig. 7: Experimental ROIs expressing dual-fluorophores  

a, Proportion of functionally defined ROIs classified as expressing one or two fluorophores based on thresholded beta multipliers. b, Frequency of dual fluorophore assignments across the dataset. Left: Heatmap showing co-assignment rates between fluorophore pairs. Right: Total frequency of dual hits per individual fluorophore. c, Frequency of dual-labeled ROIs by brain region. Left: Heatmap showing co-occurrence between projection-defined populations. Right: Total frequency of dual hits per primary brain region. (n = 1,327 ROIs) d, Example ROIs from miniscope imaging co-registered with confocal lambda stacks. ROIs are overlaid on three excitation channels (405, 561, 639 nm). e, Z-scored beta multipliers across all fluorophores for each example ROI, with above-threshold values circled. Dual fluorophores were assigned to two ROIs (28, 98), and only a single fluorophore to ROI 142. f, Spectral fingerprints for each example ROI (solid line) plotted against the average background spectrum for the animal (dashed line). ROI 28 (top) was assigned two spectrally distinct fluorophores (mTagBFP2 + mNeptune2.5); ROI 98 (middle) shows co-assignment of more spectrally overlapping fluorophores (mOrange2 + mNeptune2.5); ROI 142 (bottom) is included as a single-label example with a strong match to mTagBFP2 only. Scale bars = 10 µm.  

# Supp. Fig. 8: Neural cell types and behavioral encoding  

a, Behavioral paradigm used to identify neurons modulated during social behavior and memory. Data displayed are from the testing phase. b, Schematic of neuronal cell-types and the behavioral categories for which each cell encodes. The top tier identifies the contribution of cells from each animal subject, the second tier and color depict the neural cell-type by projection region, the third identifies the behavior in which the neurons statistically modified their activity, and the fourth determines whether the neurons increased (+) or decreased their activity (-).  

# Supp. Fig. 9: Performance of Neuroplex under reduced fluorophore complexity  

a, Schematic of experimental design: Four retrograde fluorophores were injected into distinct brain regions of GCaMP6s transgenic mice to label projection-defined pyramidal neurons in the mPFC. mTagBFP2 was injected into the c-mPFC, mVenus into the Str, mOrange2 into the Cla, and mNeptune2.5 into the VTA. b–c, Spatial distribution of identified ROIs overlaid on anatomical reference images (left) and corresponding average spectral fingerprints from each experimental animal (right). Shaded regions represent ±1.5 std from the mean. Scale bars 100 µm. d, Proportion of ROIs classified as single- or dual-labeled based on dual-pass thresholding. Animal n = 2; ROI n = 289. e, Percent of identified ROIs assigned to each individual fluorophore. f, Frequency of dual-fluorophore assignments across the dataset. Left: Heatmap showing pairwise co-occurrence rates between fluorophore (and region) combinations. Right: Total frequency of dual hits per individual fluorophore/region. g–h, Modeled experimental conditions assessing classification accuracy in either single-fluorophore (g) or dual-fluorophore (h) expression contexts. Spectra were simulated with empirical fluorophore distributions, real background, GCaMP contamination, and Gaussian white noise (left). Breakdown of error types (right). ROIs n = 460, replicates n = 100. i, Fluorophore interaction analysis showing classification accuracy when fluorophores were co-expressed. Bars indicate the percentage of ROIs correctly identified (colored), missed (gray: false negatives), or incorrectly labeled (white: false positives).  

Experimental data  

b)  

c)  

d)  

e)  

f)  

Modeled data  

Experimental conditions: Single expression  

g)  

Experimental conditions: Dual expression  

i)  

Fluorophore interaction  

h)  

Phillips et al. Multicolor neuron identification within miniscope  
