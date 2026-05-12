---
paper_id: d1c20b062d314c2cafadecb3c1f5b0c9
doi: 10.1101/2024.09.26.615039
source: biorxiv
version_date: '2025-03-17'
license: All rights reserved
title: Tissue resident memory CD4$^{+}$ T cells are sustained by site-specific levels of self-renewal and continuous replacement
authors:
- name: Jodie Chandler
  affiliations:
  - 1
  corresponding: false
  email: null
- name: M. Elise Bullock
  affiliations:
  - 2
  corresponding: false
  email: null
- name: Arpit C. Swain
  affiliations:
  - 2
  corresponding: false
  email: null
- name: Cayman Williams
  affiliations:
  - 1
  corresponding: false
  email: null
- name: Christian H. van Dorp
  affiliations:
  - 2
  corresponding: false
  email: null
- name: Benedict Seddon
  affiliations:
  - 1
  corresponding: true
  email: benedict.seddon@ucl.ac.uk
- name: Andrew J. Yates
  affiliations:
  - 2
  corresponding: true
  email: andrew.yates@columbia.edu
affiliations:
  1: Institute of Immunity and Transplantation, Division of Infection and Immunity, UCL, Royal Free Hospital, Rowland Hill
    Street, London, United Kingdom
  2: Department of Pathology and Cell Biology, Columbia University Irving Medical Center, New York, NY, USA
abstract: Tissue-resident memory T cells (T_{RM}) protect from repeat infections within organs and barrier sites. The breadth
  and duration of such protection is defined at minimum by three quantities; the rate at which new T_{RM} are generated from
  precursors, their rate of self-renewal, and their rate of loss through death, egress, or differentiation. Quantifying these
  processes individually is challenging. Here we combine genetic fate mapping tools and mathematical models to untangle these
  basic homeostatic properties of CD4^{+} T_{RM} in the skin and gut lamina propria (LP) of healthy adult mice. We show that
  CD69^{+}CD4^{+} T_{RM} in skin reside for \~24 days and self-renew more slowly, such that clones halve in size approximately
  every 5 weeks; and approximately 2% of cells are replaced daily from precursors. CD69^{+}CD4^{+} T_{RM} in LP have shorter
  residencies (\~14 days) and are maintained largely by immigration (4-6% per day). We also find evidence that the continuous
  replacement of CD69^{+}CD4^{+} T_{RM} at both sites derives from circulating effector-memory CD4^{+} T cells, in skin possibly
  via a local CD69^{-} intermediate. Our approach maps the ontogeny of CD4^{+} T_{RM} in skin and LP and exposes their dynamic
  and distinct behaviours, with continuous seeding and erosion potentially impacting the duration of immunity at these sites.
keywords:
- Tissue resident CD4$^{+}$ T cells
- mathematical modelling
- genetic fate mapping
paper_type: research-article
subject_areas:
- Immunology
- Mathematical Biology
- Cell Biology
datasets: null
stats:
  word_count: 6994
  has_math: true
  section_count: 12
---
# Tissue resident memory CD4$^{+}$ T cells are sustained by site-specific levels of self-renewal and continuous replacement

# Tissue resident memory CD4$^{+}$ T cells are sustained by site-specific levels of self-renewal and continuous replacement  

Jodie Chandler$^{1*}$, M. Elise Bullock$^{2*}$, Arpit C. Swain$^{2}$, Cayman Williams$^{1}$, Christian H. van Dorp$^{2}$, Benedict Seddon$^{1\ddagger}$, Andrew J. Yates$^{2*\ddagger}$  

*Contributed equally  

$^{1}$Institute of Immunity and Transplantation, Division of Infection and Immunity, UCL, Royal Free Hospital, Rowland Hill Street, London, United Kingdom  

²Department of Pathology and Cell Biology, Columbia University Irving Medical Center, New York, NY, USA  

$^\dagger$Address correspondence to either author; andrew.yates@columbia.edu, benedict.seddon@ucl.ac.uk  

Running title: CD4$^{+}$ T$_{\mathrm{RM}}$ dynamics in skin and lamina propria  

# Abstract  

Tissue-resident memory T cells (T_{RM}) protect from repeat infections within organs and barrier sites. The breadth and duration of such protection is defined at minimum by three quantities; the rate at which new T_{RM} are generated from precursors, their rate of self-renewal, and their rate of loss through death, egress, or differentiation. Quantifying these processes individually is challenging. Here we combine genetic fate mapping tools and mathematical models to untangle these basic homeostatic properties of CD4^{+} T_{RM} in the skin and gut lamina propria (LP) of healthy adult mice. We show that CD69^{+}CD4^{+} T_{RM} in skin reside for \~24 days and self-renew more slowly, such that clones halve in size approximately every 5 weeks; and approximately 2% of cells are replaced daily from precursors. CD69^{+}CD4^{+} T_{RM} in LP have shorter residencies (\~14 days) and are maintained largely by immigration (4-6% per day). We also find evidence that the continuous replacement of CD69^{+}CD4^{+} T_{RM} at both sites derives from circulating effector-memory CD4^{+} T cells, in skin possibly via a local CD69^{-} intermediate. Our approach maps the ontogeny of CD4^{+} T_{RM} in skin and LP and exposes their dynamic and distinct behaviours, with continuous seeding and erosion potentially impacting the duration of immunity at these sites.  

Keywords: Tissue resident CD4$^{+}$ T cells, mathematical modelling, genetic fate mapping  

# Introduction  

Resident memory T cells (T_{RM}) provide immune surveillance and protection in tissues throughout the body (Szabo et al., 2019), but the mechanisms by which they are maintained are not well understood. Conventional CD4^{+} and CD8^{+} T_{RM} in mice and humans are not intrinsically long-lived, but appear to self-renew slowly as assessed by readouts of cell division such Ki67 expression or BrdU incorporation, at levels that vary across tissues (Gebhardt et al., 2009, Watanabe et al., 2015, Park et al., 2018, Strobl et al., 2020, Divito et al., 2020, Christo et al., 2021). After infection or immune challenge, the numbers of elicited T_{RM} may also be sustained by influx from precursor populations, although the extent to which this occurs is unclear, and is likely also cell subset- and tissue-dependent. For example, in the lung there is evidence both for (Zammit et al., 2006, Ely et al., 2006, Slütter et al., 2017, Van Braeckel-Budimir et al., 2018, Takamura and Kohlmeier, 2019) and against (Takamura et al., 2016, van Dorp et al., 2024) ongoing recruitment of new T_{RM} following respiratory virus infections. Within skin, T_{RM} may be renewed or supplemented slowly from precursors in the setting of graft-versus-host disease (Divito et al., 2020), and from circulating central memory T_{CM} or effector memory T_{EM} following infection or sensitisation (Gaide et al., 2015, Matos et al., 2022). In the small intestine, however, resident CD4^{+} and CD8^{+} T_{RM} appear to persist for months to years with slow self-renewal without appreciable influx (Bartolomé-Casado et al., 2019, 2021).  

The dynamics of production and loss of $T_{\mathrm{RM}}$ in the steady state are even less well understood, and measuring these processes is important for several reasons. The balance of loss and self-renewal defines the persistence of clonal populations and hence the duration of protective immunity. Further, while self-renewal can at best preserve clonal diversity within a tissue site, any supplementation or replacement by immigrant $T_{\mathrm{RM}}$ will perturb the local TCR repertoire. In particular, any significant influx into $T_{\mathrm{RM}}$ niches in the absence of overt infection may be a competitive force, potentially reducing the persistence $T_{\mathrm{RM}}$ previously established in response to infection or challenge.  

The kinetics of circulating memory T cells have been quantified extensively in both mice and humans, using dye dilution assays (Choo et al., 2010), deuterium labelling (Westera et al., 2013, 2015, del Amo et al., 2018, Baliu-Piqué et al., 2018, 2019, van den Berg et al., 2021), and BrdU labelling, either alone (Younes et al., 2011, Ganusov and De Boer, 2013) or in combination with fate reporters (Gossel et al., 2017, Hogan et al., 2019, Bullock et al., 2024) or T cell receptor excision circles (den Braber et al., 2012). Using mathematical models to interpret these data, these studies identified rates of production, cellular lifespans, and signatures of heterogeneity in turnover. modelling has also established evidence for continuous replenishment of circulating memory CD4$^{+}$ T cells from precursors throughout life in specific-pathogen-free mice, driven by a combination of environmental, commensal and self antigens (Gossel et al., 2017, Hogan et al., 2019, Bullock et al., 2024). Quantification of T$_{\text{RM}}$ dynamics has to date been restricted largely to measuring the net persistence of CD8$^{+}$ T$_{\text{RM}}$ following infection in mice, in a variety of tissues (Morris et al., 2019, Wijeyesinghe et al., 2021). Far less is known regarding CD4$^{+}$ T$_{\text{RM}}$, which typically outnumber their CD8$^{+}$ counterparts (Szabo et al., 2019), and there have been very few attempts to dissect the kinetics of either subset (van Dorp et al., 2024).  

In general, measuring these basic parameters in isolation is challenging, partly due to their sensitivity to assumptions made in the models (De Boer and Perelson, 2013), but also because division-linked labelling alone may not distinguish in situ cell division and the supplementation of a population from labelled precursors. A more powerful approach is to triangulate information from different readouts of cell fate simultaneously (Bains et al., 2009, den Braber et al., 2012, del Amo et al., 2018, De Boer and Yates, 2023, Bullock et al., 2024).  

With these challenges in mind, here we integrated data from two independent inducible fate reporter systems to study CD4$^{+}$ T$_{\text{RM}}$ homeostasis in mice. Each system allows one to track the fates of defined populations of cells and their descendants. One labels all CD4$^{+}$ T cell subsets at any given moment, which effectively provides an age 'timestamp'. The other labels cells that are dividing during a defined time window. In combination, these systems allowed us to establish a quantitative model of the basal homeostatic properties of CD4$^{+}$ T$_{\text{RM}}$ within the skin and the lamina propria of the small intestine in healthy mice. In particular, we could unpick the contributions of self renewal and de novo cell production that underpin their maintenance, and explore their relationships to circulating T cell subsets.  

# Results  

# Combining cell fate reporters and models to measure $T_{\mathrm{RM}}$ replacement, loss, and self-renewal  

To study the homeostatic dynamics of tissue resident CD4$^{+}$ memory T cells in healthy mice, we used in concert two genetic fate mapping tools in which cohorts of peripheral T cells and their offspring can be induced to express permanent fluorescent markers (Fig. 1A). These reporter strains were previously used separately to study the turnover of naive and circulating memory T and B cells (Verheijen et al., 2020, Lukas et al., 2023, Bullock et al., 2024). In the Ki67$^{\text{mCherry-CreERT}}$ Rosa26$^{\text{RcagYFP}}$ system, henceforth Ki67-DIVN, fluorescent reporters are linked to the expression of Ki67, a nuclear protein that is expressed during cell division and for 3 to 4 days afterwards (Gossel et al., 2017, Miller et al., 2018). Specifically, these mice express both a Ki67-mCherry fusion protein and inducible CreERT from the Mki67 locus, together with a Rosa26$^{\text{RcagYFP}}$ Cre reporter construct. Treatment of mice with tamoxifen therefore induces YFP in cells expressing high levels of Ki67, and YFP is then stably expressed by these cells and their offspring. Expression of Ki67-fused mCherry gives a constitutive live readout of Ki67 expression, independent of tamoxifen treatment and YFP expression. In the second fate reporter, CD4$^{\text{CreERT}}$ Rosa26$^{\text{RnTom}}$ mice (Cd4-FR), the Cre reporter is constructed such that cells expressing CD4 during tamoxifen treatment permanently and heritably express the fluorescent reporter mTomato (mTom).  

In a closed population of cells at steady state, self-renewal must be balanced by loss and so, following tamoxifen treatment, the frequencies of cells expressing YFP or mTom within any such population would remain constant. Therefore, any decline in the frequency of either reporter among $T_{\mathrm{RM}}$ after treatment must derive from the influx of label-negative cells from an upstream (precursor) population. In the Ki67-DIVN mice, these will be descendants of cells that were not dividing at the time of tamoxifen treatment; in the Cd4-FR mice, labelled CD4$^{+}$ cells will slowly be replaced by the descendants of those generated in the thymus after treatment. The shape of this decline will be determined by the combination of the net loss rate of $T_{\mathrm{RM}}$ from the tissue (through death, egress, or differentiation, offset by any self-renewal), and the label content of immigrant $T_{\mathrm{RM}}$ (Fig. 1B). To refer to the persistence of individual $T_{\mathrm{RM}}$ cells we will use the term 'residence time' rather than lifespan, to reflect the multiple potential mechanisms of loss from tissues.  

To quantify these processes, we treated cohorts of both reporter mice, aged between 4 and 15 weeks, with a single 2mg pulse of tamoxifen (Fig. 1C). Over 9 week (Ki67 reporter) and 57 week (CD4 reporter) chase periods we measured the frequencies of labelled cells among antigen-experienced CD4$^{+}$ T cell subsets isolated from skin and the lamina propria of the small intestine (henceforth LP), and within circulating naive and memory T cell subsets derived from lymph nodes (Fig. S1A). By combining these frequencies with measures of Ki67 expression, and describing the resulting set of time series with simple mathematical models, we aimed to estimate the basic parameters underlying T$_{\text{RM}}$ kinetics.  

# CD4$^{+}$ T$_{RM}$ in skin and lamina propria are continuously replaced from precursors  

We considered two populations within both skin and LP, identified as tissue-localised by virtue of their protection from short-term in vivo labelling (Methods; Fig. S1B). One was effector-memory (EM) phenotype (CD4$^{+}$CD44$^{\mathrm{hi}}$ CD62L$^{\mathrm{lo}}$) T cells in bulk, which we studied in order to gain the broadest possible picture of memory T cell dynamics at these sites. We also considered the subset of these cells that expressed CD69, a canonical and consistent marker of CD4$^{+}$ T cell residency across multiple tissues (Szabo et al., 2019). We saw no significant changes with mouse age in the numbers of either population within skin (Fig. 2A, $p > 0.67$) or LP (Fig. 2B, $p > 0.39$). There were also no significant changes in any of these quantities with time since tamoxifen treatment ($p > 0.24$). We therefore assumed that the skin- and LP-localised T cell subsets we considered were at, or close to, homeostatic equilibrium during the chase period. For brevity, we refer to tissue-localised CD4$^{+}$CD44$^{\mathrm{hi}}$CD62L$^{\mathrm{lo}}$ in bulk as EM, and their CD69$^{+}$ subset as $\mathbf{T}_{\mathrm{RM}}$.  

During the first few days after tamoxifen treatment, YFP and mTom expression increased continuously within the skin and LP subsets (Fig. 2C), as well as among CD4$^{+}$ naive (CD44$^{lo}$ CD62L$^{hi}$), central memory (T$_{CM}$, CD44$^{hi}$  

C Label kinetics within target (tissue-localised) populations  

D Label kinetics within candidate precursor populations  

CD62L$^{\mathrm{hi}}$) and effector memory (T$_{\mathrm{EM}}$, CD44$^{\mathrm{hi}}$ CD62L$^{\mathrm{lo}}$) T cells recovered from lymph nodes (Fig. 2D). These initial increases were driven in part by the intracellular dynamics of the induction of the fluorescent reporters. We therefore began our analyses at day 5 post-treatment, by which time induction was considered complete and the subsequent trajectories of label frequencies reflected only the dynamic processes of cell production and loss. A key observation was that mTom expression within the skin and LP subsets then declined slowly (roughly 7- to 8-fold over the course of a year, Fig. 2C), indicating immediately that these populations were being continuously replaced from precursors. Early in the chase period YFP$^{+}$ T$_{\mathrm{RM}}$ expressed Ki67 at higher levels than YFP$^{-}$ cells, as expected, but Ki67 expression in the two populations converged at later times. We return to the interpretation of these kinetics below.  

We then investigated the extent to which the simple model illustrated in Fig. 1B could explain these trajectories. Given the observation of continued recruitment, any time-variation in the label content of $T_{\mathrm{RM}}$ precursors might leave an imprint on the label kinetics of skin or LP $T_{\mathrm{RM}}$ themselves, and thereby help us to identify their developmental pathways. We reasoned that plausible $T_{\mathrm{RM}}$ precursors might be LN-derived CD4$^{+}$ naive, $T_{\mathrm{CM}}$ or $T_{\mathrm{EM}}$; we also considered the possibilities that CD69$^{-}$ cells within skin and LP are the direct precursors of the local CD69$^{+}$ populations. Therefore, we used empirical functions to describe the timecourses of the frequencies of YFP$^{+}$ and mTom$^{+}$ cells within these populations (Fig. 2D), and used these to represent the label composition of cells entering the tissue subsets.  

For each tissue subset ('target') and precursor pair, we fitted the model simultaneously to six timecourses; the frequencies of (i) YFP expression and (ii) mTom expression among target cells, the proportions of Ki67$^{high}$ cells among (iii) YFP$^{+}$ and (iv) YFP$^{-}$ target cells, and the (v) YFP and (vi) mTom expression kinetics within the precursor (Methods, and Supporting Information Text S1). For each precursor/target pair we considered three modes of influx – one in which new immigrant $T_{RM}$ are Ki67$^{low}$ ('quiescent' recruitment); another in which their Ki67 expression directly reflects that of the precursor ('neutral' recruitment), and a third in which immigrants have recently divided (Ki67$^{high}$), perhaps through an antigen-driven process ('division-linked' recruitment).  

# Skin and LP CD4$^{+}$ T$_{\text{RM}}$ have similar residence times but exhibit distinct contributions of replacement and self-renewal  

For each combination of target population, potential precursor, and potential mode of recruitment, we were able to estimate rates of influx, mean residence times, and mean interdivision times for the target population (Fig. 3 and Table S1; prior and posterior distributions of the parameters of the best fitting models are shown in Fig. S2). The mean residence times of both EM and $T_{\mathrm{RM}}$ within skin and LP were $\sim$3 weeks and 2 weeks respectively. The means of production of new cells differed at the two sites, however. In skin, around 2% of both populations were replaced daily by influx, comparable to the rates of constitutive replacement of circulating memory CD4$^{+}$ T cell subsets (Gossel et al., 2017, Hogan et al., 2019, Bullock et al., 2024), and EM and $T_{\mathrm{RM}}$ self-renewed every 6 and 7 weeks respectively, In contrast, within LP these subsets divided less often (every 7-9 weeks) and relied on higher levels of recruitment (4-6% per day) for their maintenance.  

From these basic quantities we could derive several other important measures of $T_{\mathrm{RM}}$ behaviour. First, the balance of the rates of loss ($\delta$) and self-renewal ($\rho$) defines the persistence of a cohort of T cells, which is distinct from the lifespan of its constituent cells (del Amo et al., 2018, De Boer and Yates, 2023). Specifically, the quantity $\ln(2)/(\delta - \rho)$ is the average time taken for a cohort and their descendents to halve in number. While we studied polyclonal populations here, this quantity applies equally well to measuring the persistence of a TCR clonotype, so we refer to it as a clonal half life (Bullock et al., 2024). The substantial rates of self-renewal in skin led to clonal half lives of just over a month. The lower levels of self-renewal and higher levels of replacement in LP resulted in shorter clonal half lives of $\sim$2 weeks.  

Importantly, our estimates of these quantities depended to varying degrees on the choice of precursor and mode of recruitment (Fig. 3). For example, intuitively, given the observed level of Ki67 within each target population, the greater the levels of Ki67 within newly recruited cells, the less must derive from self-renewal within the tissue; hence, if one assumes that recruitment is division-linked, estimated division rates are reduced. Similarly, as discussed above,  

the label content of the precursor influences the net loss rate of label in the target, which was most clearly reflected in the loss of mTom$^{+}$ cells over the longer chase period (Fig. 2C and D). For example, mTom was lost most rapidly within naive CD4$^{+}$ T cells (Fig. 2D), due to export of label-negative cells from the thymus. Models in which naive T cells were the direct precursor of T$_{\text{RM}}$ therefore predicted greater clonal persistence within tissues.  

We saw very little decline in YFP expression levels during the 2 month chase period (Fig. 2C), due in part to the sustained levels of YFP within the putative precursor populations (Fig. 2D), which 'topped up' YFP-expressing $T_{\mathrm{RM}}$. As a result, YFP kinetics within the tissues was not strongly informative regarding rates of replacement. However, the rate of convergence of Ki67 within YFP$^{+}$ and YFP$^{-}$ cells (Fig. 2C) put clear constraints on the duration of Ki67 expression, which at approximately 3 days (Fig. 3) was consistent with previous estimates. This quantity in turn was informative for estimating rates of self-renewal. Further, the observed convergence of Ki67 within these two subsets is consistent with the basic model assumption of homogeneity in the rates of division and loss within skin and LP.  

CD4$^{+}$ CD69$^{+}$ T$_{RM}$ within LP likely derive predominantly from circulating T$_{EM}$ in lymph nodes, while those in skin may derive from a CD69$^{-}$ intermediate  

To more precisely quantify the kinetics of each target population, we assessed the relative support for each combination of precursor and mode of recruitment (Fig. 4A). Each of these weights summarises the magnitude and uncertainty of a model's out-of-sample prediction error of the label kinetics within the target population (Methods).  

We found that the data were quite strongly informative regarding the immediate ancestors of tissue subsets. From the candidate set of models, the weighting strongly favoured lymph-node derived EM as the closest precursor to CD4$^{+}$  

EM within both skin and LP. However, within skin, local CD69$^{-}$ cells were the favoured precursor to CD69$^{+}$ T$_{\text{RM}}$. The evidence was generally more equivocal regarding the mode in which cells are recruited into skin and LP, although for skin we saw substantial evidence (66% of model support) for a division-linked transition from CD69$^{-}$ to CD69$^{+}$ cells. Fig. 4B summarises the developmental trajectories and kinetics of T$_{\text{RM}}$ in skin and LP that were supported most strongly by our analyses. Parameter estimates and credible intervals for these models are highlighted with vertical shaded regions in Fig. 3 and are detailed in Supporting Information, Table S1. Visual differences between models are shown in Fig. S3, where for each target population we overlay the fits from top-ranked, second-ranked, and lowest-ranked models.  

# Validation of residence times through use of Ki67 expression directly  

As a consistency check, when a population is at or close to steady state, bounds on the mean residence time of cells can be estimated using only the measured frequency of Ki67 within the target population, and the daily rate of replacement (Supporting Information, Text S2). In skin, both EM and $T_{\mathrm{RM}}$ are replaced at the rate of 2% per day, and have Ki67 expression frequencies of around 0.15 (Fig. 2C). The approximation then yields residence times in the range 22-28 days, depending on whether immigrant $T_{\mathrm{RM}}$ are $Ki67^{\mathrm{low}}$ or $Ki67^{\mathrm{high}}$ respectively. In LP, with 5.5% daily replacement and Ki67 frequencies of around 0.07, we estimate residence times of 14-25 days. Both estimates are in good agreement with those from the model fitting. Further validation of these results, and dissection of the kinetics, might be achieved by manipulating cell trafficking, although this would potentially impact multiple processes at once. For example, treating mice with the sphingosine 1-phosphate receptor agonist FTY720 would block tissue ingress and egress. This would leave self-renewal as the only means of $T_{\mathrm{RM}}$ production, and would also remove the component of the cell loss rate that is due to cells leaving the tissue. In principle one could then gain estimates of the intrinsic lifespan of $T_{\mathrm{RM}}$, rather than their tissue residence time. However, parameter estimation would then require accurate measurements of cell numbers within the tissue.  

# Discussion  

Our analysis indicates that CD4$^{+}$ T$_{\text{RM}}$ are not intrinsically long-lived, but instead are sustained by both self-renewal and supplementation from circulating precursors. By combining fate reporting methods with mathematical models, we also showed that it is possible to separately quantify the processes that underlie their persistence. We saw quite distinct contributions of recruitment and self-renewal of both subsets within skin and LP. The basis of this difference is unclear, but we speculate that the large antigenic burden within the small intestine drives the higher levels of T$_{\text{RM}}$ recruitment and clonal erosion within the lamina propria. We showed that estimates of these quantities depend on the identity of any precursor, whose label kinetics propagate downstream into the population of interest; and the extent of any cell division that occurs around the time of differentiation or ingress. However, by using easily interpretable mathematical models, and assessing the statistical support for each, we were able to measure the support for different pathways and modes of recruitment into each subset.  

The schematic in Fig. 1B illustrates a hypothetical example in which the frequency of YFP-expressing cells within a precursor declines. This trend is then reflected downstream in the target. However, in our experiments the kinetics of YFP in most of the putative precursors were quite flat (Fig. 2D). As noted above, these kinetics were likely due largely to the continued influx of new cells into circulating memory subsets, likely from naive precursors (Gossel et al., 2017, Hogan et al., 2019, Bullock et al., 2024). These had increasing levels of YFP (Fig. 2D, left panel) deriving from thymocytes that were dividing rapidly during treatment. YFP levels among naive T cells were also likely sustained to an extent by low-level residual labelling of thymic progenitors (Lukas et al., 2023). YFP expression was therefore not cleanly 'washed out' in the periphery. The data from the labelling of CD4-expressing cells were more informative for dissecting turnover; mTom⁺ cells were clearly diluted out of all peripheral populations by the descendants of mTom⁻ thymocytes.  

In these reporter mice, YFP and mTom were induced quickly in all subsets to different degrees; therefore our inferences regarding precursor-target relationships weren't informed by the initial levels of label in each. (For example,  

imagine a rapidly dividing target fed by a slowly dividing precursor; initially, YFP levels in the target would be higher than those in the precursor). The hierarchy of levels of label in different subsets would be informative if one expects targets to begin with no label at all; for instance, in the busulfan chimeric mouse system (Hogan et al., 2015) new, thymically derived 'labelled' (donor) cells progressively infiltrate replete 'unlabelled' (host) populations. In that case, one can immediately reject certain differentiation pathways by examining the sequence of accrual of donor cells in different subsets. In the systems we use here, information regarding lineage relationships is contained instead in the trends in YFP and mTom frequencies after treatment, because precursor kinetics must leave an imprint on the target (Fig. 1B). This information is particularly useful if two populations exhibit opposing trajectories – they are then unlikely to be immediately related.  

On a technical note, in general one can reduce variation by comparing quantities derived from the same individual. We showed previously that in some situations, exploiting the within-mouse grouping of observations can reduce uncertainty and refine parameter estimates when modelling the dynamics of cell populations (Yates et al., 2007). We were not able to take this approach here, however, because the frequency of cells expressing YFP or mTom within a given subset in a particular mouse depends on the accumulated history of label in that subset's precursor. We were unable to identify these trajectories for each animal, so we were obliged to use the population averages (Fig. 2D). To mitigate any biases this averaging might introduce, we fitted these empirical functions simultaneously with the models of label kinetics in the targets. This conservative strategy propagated the uncertainty in the precursor trajectories into our conclusions.  

Our analysis was rooted in the observation that all CD4$^{+}$ T cell subsets within these tissues were at steady-state. This dynamic equilibrium dictates that immigration of new cells must be accompanied by loss of existing ones. In the SPF mice studied here, these dynamic populations are likely specific for self or commensal antigens that are continuously expressed. It is possible that residence and interdivision times are distinct for T$_{\text{RM}}$ that might not be replenished long-term from precursors, such as those generated in acute infections. Further, our simple model can explain the stable maintenance of T$_{\text{RM}}$ numbers in healthy skin and LP without needing to invoke the concept of a homeostatic niche, such as a competitive limit to cell densities. In this model, any increase or decrease in the rate of influx into a tissue will simply lead to a new equilibrium at higher or lower cell densities, respectively. Indeed repeated vaccinia virus challenges can drive progressive increases in the numbers of virus-specific CD8$^{+}$ T$_{\text{RM}}$ in skin that are detectable for months (Jiang et al., 2012), and heterologous challenges appear not to erode pre-existing LCMV-specific CD8$^{+}$ T$_{\text{RM}}$ (Wijeyesinghe et al., 2021). However, whether the same flexibility manifests among CD4$^{+}$ T$_{\text{RM}}$ following repeated challenges is unclear.  

Our goal here was to assess the support for external replenishment of effector-memory-like CD4$^{+}$ T cells in bulk, and the dominant CD69$^{+}$ T$_{\text{RM}}$ subset. We found evidence that CD69$^{+}$CD4$^{+}$ T$_{\text{RM}}$ in skin derive at least in part from a local CD69$^{-}$ precursor. With richer phenotyping of tissue-localised cells, we could in principle use labelling trajectories to define more fine-grained differentiation pathways. One issue is that accurate measurement of label frequencies becomes more difficult as one resolves T$_{\text{RM}}$ into smaller subsets; indeed we saw that label kinetics within the small CD69$^{-}$ populations were relatively noisy. Another issue is that label kinetics operate on the timescales of the net loss rate of cell populations – death and onward differentiation, balanced by self-renewal. More frequent sampling would be required to resolve more transitory intermediates. Nevertheless, our study clearly exposes the highly dynamic nature of CD4$^{+}$ T$_{\text{RM}}$, sustained throughout life by both self renewal and continued influx from precursors. This tissue-specific influx, particularly if there are competitive limits to T$_{\text{RM}}$ occupancy, may contribute to the differential longevity of immunity at different barrier sites.  

# Acknowledgements  

This work was supported by the National Institutes of Health (R01 AI093870 and U01 AI150680) and the Medical Research Council (MR/P011225/1).  

# Methods  

Reporter mouse strains Ki67$^{m}$Cherry-CreERT Rosa26$^{RcagYFP}$ (Ki67-DIVN) and CD4$^{CreERT}$ Rosa26$^{RmTom}$ (Cd4-FR) mice have been described previously (Bullock et al., 2024). Experimental Ki67-DIVN mice were homozygous for indicated mutations at both the Ki67 and Rosa26 loci. Experimental CD4-FR mice were heterozygous for the indicated mutations and both the CD4 and Rosa26 loci. Tamoxifen (Sigma) was diluted to 20mg/mL in corn oil (Fisher Scientific) and 100$\mu$l (2mg) was administered to mice via oral feeding on day 0. Ki67-DIVN mice were injected with 2$\mu$g Thy1.2-BV510 (53-2.1) (BioLegend) 3 minutes prior to sacrifice to label T cells in the circulation. This protocol typically achieves >99% staining of circulating cells (Anderson et al., 2014) and less than 3% of cells recovered from our tissue samples were label positive (Fig. S1B), supporting the assumption of very low rates of false positive and false negative events. Mice were subsequently taken down at specified timepoints post tamoxifen treatment for organ collection.  

Cell preparation All peripheral lymph nodes (LN), the small intestine (SI) and ear skin were taken from mice and processed into single cell suspensions. LNs were mashed through two pieces of fine gauze in a petri dish and washed with complete RPMI (ThermoFisher) supplemented with 5% FCS (ThermoFisher) (cRPMI). Cells were resuspended in cold PBS and counted using the CASY counter (Cambridge Bioscience). Peyer's patches were excised from the antimesenteric side of the SI before being opened longitudinally and SI contents scraped out. SI pieces were placed in 20mL pre-warmed extraction media (cRPMI + 10mM HEPES (ThermoFisher) + 5mM EDTA (Sigma) + 1mM DTT (Abcam)) and incubated in 37°C shaking incubator for 30 minutes at 200rpm. Cells were filtered over 70μm cell strainer (supernatant containing intra-epithelial lymphocytes not used), and SI pieces were placed in cold cRPMI supplemented with 10mM HEPES and allowed to settle. Supernatant was carefully poured off and SI pieces were finely minced, added to 20mL pre-warmed digestion media (RPMI + 10% FCS + 1.5mg/mL collagenase VIII (Sigma)) and incubated in 37°C shaking incubator for 30 minutes at 200rpm. After digestion, cells were passed through 70μm cell strainer and washed with cRPMI + 10mM HEPES. The resulting cell suspension contains cells from the lamina propria (LP) of the SI. Ear skin was excised and separated into dorsal and ventral sides. Skin was finely minced, added to 4mL digestion buffer (cRPMI + 50mM HEPES + 37.5μg/mL Liberase TL (Merck)+ 3.125mg/mL collagenase IV (ThermoFisher)+ 1mg/mL DNase I (Merck)) and incubated in 37°C shaking incubator for 2 hours at 200rpm. Cells were filtered over 70μm cell strainer and washed through with cRPMI.  

Flow cytometry All cells isolated from skin and LP, and $5 \times 10^{6}$ LN cells, were stained for analysis by flow cytometry. Cells were stained in 100$\mu$l PBS with combinations of: CD8$\alpha$-BUV395 (53-6.7), CD25-BUV395 (PC61), CD62L-BUV737 (MEL-14), TCR$\gamma\delta$-BV421 (GL3), CD103-BV786 (M290) (all BD Biosciences); CD25-BV650 (PC61), CD103-BV421 (2E7), CD8$\alpha$-BV570 (53-6.7), TCR$\gamma\delta$-BV605 (GL3), NK1.1-BV650 (PK136), CD44-BV785 (IM7), CD45.1-BV605 (A20), CD4-BV711 (RM4-5), CD8b.2-APC (53-5.8) (all BioLegend); TCR$\beta$-PerCPCy5.5 (104) (Cambridge Bioscience); CD44-APCef780 (IM7) (eBioscience); and CD3-APCef780 (2C11), CD3-biotin (145-2C11), CD69-PeCy7 (H1.2F3), CD45.2-AF700 (104), nearIR live/dead, blue live/dead, yellow live/dead (all ThermoFisher). Cells were fixed for 20 minutes with IC fix (Invitrogen) and washed twice in FACS buffer (PBS + 0.1% BSA). Flow cytometric analysis was performed on either a Cytek Aurora spectral flow cytometer or a conventional BD LSR-Fortessa and analysed using FlowJo software (Treestar).  

Cell count calculations Cell counts of LN populations were calculated by dividing the event count in a target population by the event count of live cells, multiplied by the total live cells in LN prep determined by CASY counter. Sizes of LP and skin populations were calculated using AccuCount (Spherotech) counting beads that were spiked into the sample prior to acquisition, as per manufacturer's instructions.  

Mathematical modelling and Statistical Analysis We fitted simultaneously the kinetically homogeneous mathematical model illustrated in Fig. 1B, and described in Supporting Information Text S1, to the time courses of frequencies of YFP$^{+}$, Ki67$^{\text{high}}$ in YFP$^{+}$, Ki67$^{\text{high}}$ in YFP$^{-}$, and mTom$^{+}$ cells in the target populations; and empirical  

descriptor functions describing the trajectories of the frequencies of YFP$^{+}$ and mTom$^{+}$ precursors (all data shown in Fig. 2C). We used a Bayesian estimation approach using Python and Stan (Stan Development Team, 2024) to perform these model fits. Code and data used to perform model fitting, details of the prior distributions for parameters, and figure generation notebooks are available at https://github.com/elisebullock/CD4TRM. Models were ranked based on information criteria estimated using the Leave-One-Out (LOO) cross validation method (Vehari et al., 2017). See Supporting Information, Text S1 for full details.  

# References  

Anderson KG, Mayer-Barber K, Sung H, et al. (2014). Intravascular staining for discrimination of vascular and tissue leukocytes. Nat Protoc 9(1):209–22

Bains I, Thiébaut R, Yates AJ, Callard R (2009). Quantifying thymic export: combining models of naive T cell proliferation and TCR excision circle dynamics gives an explicit measure of thymic output. J Immunol 183(7):4329–36

Baliu-Piqué M, Otto SA, Borghans JAM, Tesselaar K (2019). In vivo deuterium labelling in mice supports a dynamic model for memory T-cell maintenance in the bone marrow. Immunol Lett 210:29–32

Baliu-Piqué M, Verheij MW, Drylewicz J, et al. (2018). Short lifespans of memory T-cells in bone marrow, blood, and lymph nodes suggest that T-cell memory is maintained by continuous self-renewal of recirculating cells. Front Immunol 9:2054

Bartolomé-Casado R, Landsverk OJB, Chauhan SK, et al. (2019). Resident memory CD8 T cells persist for years in human small intestine. J Exp Med 216(10):2412–2426

Bartolomé-Casado R, Landsverk OJB, Chauhan SK, et al. (2021). CD4+ T cells persist for years in the human small intestine and display a TH1 cytokine profile. Mucosal Immunol 14(2):402–410

Bullock ME, Hogan T, Williams C, et al. (2024). The dynamics and longevity of circulating CD4+ memory T cells depend on cell age and not the chronological age of the host. PLoS Biol 22(8):e3002380

Choo DK, Murali-Krishna K, Antia R, Ahmed R (2010). Homeostatic turnover of virus-specific memory CD8 T cells occurs stochastically and is independent of CD4 T cell help. J Immunol 185(6):3436–44

Christo SN, Evrard M, Park SL, et al. (2021). Discrete tissue microenvironments instruct diversity in resident memory T cell function and plasticity. Nat Immunol 22(9):1140–1151

De Boer RJ, Perelson AS (2013). Quantifying T lymphocyte turnover. J Theor Biol 327:45–87

De Boer RJ, Yates AJ (2023). Modeling T Cell Fate. Annu Rev Immunol 41:513–532

del Amo PC, Benyeetz JL, Boelen L, et al. (2018). Human TSCM cell dynamics in vivo are compatible with long-lived immunological memory and stemness. PLoS Biology 16(6):1–22

den Braber I, Mugwagwa T, Vrisekoop N, et al. (2012). Maintenance of peripheral naive T cells is sustained by thymus output in mice but not humans. Immunity 36(2):288–97

Divito SJ, Aasebø AT, Matos TR, et al. (2020). Peripheral host T cells survive hematopoietic stem cell transplantation and promote graft-versus-host disease. J Clin Invest 130(9):4624–4636

Ely KH, Cookenham T, Roberts AD, Woodland DL (2006). Memory T cell populations in the lung airways are maintained by continual recruitment. J Immunol 176(1):537–543

Gaide O, Emerson RO, Jiang X, et al. (2015). Common clonal origin of central and resident memory T cells following skin immunization. Nature Medicine 21(6):647–653

Ganusov VV, De Boer RJ (2013). A mechanistic model for bromodeoxyuridine dilution naturally explains labelling data of self-renewing T cell populations. J R Soc Interface 10(78):20120617  

Gebhardt T, Wakim LM, Eidsmo L, et al. (2009). Memory T cells in nonlymphoid tissue that provide enhanced local immunity during infection with herpes simplex virus. Nat Immunol 10(5):524–30

Gossel G, Hogan T, Cownden D, Seddon B, Yates AJ (2017). Memory CD4 T cell subsets are kinetically heterogeneous and replenished from naive T cells at high levels. eLife 6:e23013

Hogan T, Gossel G, Yates AJ, Seddon B (2015). Temporal fate mapping reveals age-linked heterogeneity in naive T lymphocytes in mice. Proc Natl Acad Sci U S A 112(50):E6917–26

Hogan T, Nowicka M, Cownden D, et al. (2019). Differential impact of self and environmental antigens on the ontogeny and maintenance of CD4+ T cell memory. eLife 8:e48901

Jiang X, Clark RA, Liu L, et al. (2012). Skin infection generates non-migratory memory CD8+ TRM cells providing global skin immunity. Nature 483(7388):227–231

Lukas E, Hogan T, Williams C, Seddon B, Yates AJ (2023). Quantifying cellular dynamics in mice using a novel fluorescent division reporter system. Front Immunol 14:1157705

Matos TR, Gehad A, Teague JE, et al. (2022). Central memory T cells are the most effective precursors of resident memory T cells in human skin. Sci Immunol 7(70):eabn1889

Miller I, Min M, Yang C, et al. (2018). Ki67 is a graded rather than a binary marker of proliferation versus quiescence. Cell Rep 24(5):1105–1112.e5

Morris SE, Farber DL, Yates AJ (2019). Tissue-Resident Memory T Cells in Mice and Humans: Towards a Quantitative Ecology. J Immunol 203(10):2561–2569

Park SL, Zaid A, Hor JL, et al. (2018). Local proliferation maintains a stable pool of tissue-resident memory T cells after antiviral recall responses article. Nature Immunology 19(2):183–191

Slütter B, Van Braeckel-Budimir N, Abboud G, et al. (2017). Dynamics of influenza-induced lung-resident memory T cells underlie waning heterosubtypic immunity. Sci Immunol 2(7):eaag2031

Stan Development Team (2024). Stan Modeling Language Users Guide and Reference Manual, version 2.34

Strobl J, Pandey RV, Krausgruber T, et al. (2020). Long-term skin-resident memory T cells proliferate in situ and are involved in human graft-versus-host disease. Sci Transl Med 12(570):eabb7028

Szabo PA, Miron M, Farber DL (2019). Location, location, location: Tissue resident memory T cells in mice and humans. Sci Immunol 4(34)

Takamura S, Kohlmeier JE (2019). Establishment and maintenance of conventional and circulation-driven lung-resident memory CD8+ T cells following respiratory virus infections. Front Immunol 10:733

Takamura S, Yagi H, Hakata Y, et al. (2016). Specific niches for lung-resident memory CD8+ T cells at the site of tissue regeneration enable CD69-independent maintenance. J Exp Med 213(13):3057–3073

Van Braeckel-Budimir N, Varga SM, Badovinac VP, Harty JT (2018). Repeated antigen exposure extends the durability of influenza-specific lung-resident memory CD8+ T cells and heterosubtypic immunity. Cell Reports 24(13):3374–3382

van den Berg SPH, Derksen LY, Drylewicz J, et al. (2021). Quantification of T-cell dynamics during latent cytomegalovirus infection in humans. PLoS Pathog 17(12):e1010152

van Dorp CH, Gray JI, Paik DH, Farber DL, Yates AJ (2024). A variational deep-learning approach to modeling memory T cell dynamics. bioRxiv, DOI 10.1101/2024.07.08.602409

Vehtari A, Gelman A, Gabriy J (2017). Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. Stat Comput 27:1413–1432  

Verheijen M, Rane S, Pearson C, Yates AJ, Seddon B (2020). Fate mapping quantifies the dynamics of B cell development and activation throughout life. Cell Reports 33(7):108376

Watanabe R, Gehad A, Yang C, et al. (2015). Human skin is protected by four functionally and phenotypically discrete populations of resident and recirculating memory T cells. Science Translational Medicine 7(279)

Westera L, Drylewicz J, den Braber I, et al. (2013). Closing the gap between T-cell life span estimates from stable isotope-labeling studies in mice and humans. Blood 122(13):2205–12

Westera L, van Hoeven V, Drylewicz J, et al. (2015). Lymphocyte maintenance during healthy aging requires no substantial alterations in cellular turnover. Aging Cell 14(2):219–27

Wijeyesinghe S, Beura LK, Pierson MJ, et al. (2021). Expansible residence decentralizes immune homeostasis. Nature 592(7854):457–462

Yates A, Graw F, Barber DL, et al. (2007). Revisiting estimates of CTL killing rates in vivo. PLoS One 2(12):e1301

Younes SA, Punkosdy G, Caucheteux S, et al. (2011). Memory phenotype CD4 T cells undergoing rapid, nonburst-like, cytokine-driven proliferation can be distinguished from antigen-experienced memory cells. PLoS Biol 9(10):e1001171

Zammit DJ, Turner DL, Klonowski KD, Lefrançois L, Cauley LS (2006). Residual antigen presentation after influenza virus infection affects CD8 T cell activation and migration. Immunity 24(4):439–449  
