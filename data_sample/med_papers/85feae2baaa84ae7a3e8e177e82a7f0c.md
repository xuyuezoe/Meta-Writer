---
paper_id: 85feae2baaa84ae7a3e8e177e82a7f0c
doi: 10.1101/2024.07.25.605188
source: biorxiv
version_date: '2025-01-02'
license: null
title: Direct RNA sequencing enables improved transcriptome assessment and tracking of RNA modifications for medical applications
authors:
- name: Charlotte Hewel
  affiliations:
  - 1
  corresponding: false
  email: null
- name: Anna Wierczeiko
  affiliations:
  - 1
  corresponding: false
  email: null
- name: Johannes Miedema
  affiliations:
  - 1
  corresponding: false
  email: null
- name: Felix Hofmann
  affiliations:
  - 1
  corresponding: false
  email: null
- name: Stephan Weißbach
  affiliations:
  - 1
  corresponding: false
  email: null
- name: Vincent Dietrich
  affiliations:
  - 1
  corresponding: false
  email: null
- name: Johannes Friedrich
  affiliations:
  - 1
  corresponding: false
  email: null
- name: Laura Holthöfer
  affiliations:
  - 1
  corresponding: false
  email: null
- name: Verena Haug
  affiliations:
  - 1
  corresponding: false
  email: null
- name: Stefan Mündnich
  affiliations:
  - 7
  corresponding: false
  email: null
- name: Lukas Schartei
  affiliations:
  - 3
  - 4
  corresponding: false
  email: null
- name: Kristi Jenson
  affiliations:
  - 2
  corresponding: false
  email: null
- name: Stefan Diederich
  affiliations:
  - 1
  corresponding: false
  email: null
- name: Stanislav Sys
  affiliations:
  - 1
  corresponding: false
  email: null
- name: Tamer Butto
  affiliations:
  - 7
  corresponding: false
  email: null
- name: Norbert W Paul
  affiliations:
  - 9
  corresponding: false
  email: null
- name: Jonas Koch
  affiliations:
  - 11
  - 12
  corresponding: false
  email: null
- name: Frank Lyko
  affiliations:
  - 11
  - 13
  corresponding: false
  email: null
- name: Florian Kraft
  affiliations:
  - 8
  corresponding: false
  email: null
- name: Susann Schweiger
  affiliations:
  - 1
  corresponding: false
  email: null
- name: Edward A Lemke
  affiliations:
  - 3
  - 5
  corresponding: false
  email: null
- name: Mark Helm
  affiliations:
  - 7
  corresponding: false
  email: null
- name: Matthias Linke
  affiliations:
  - 1
  corresponding: false
  email: null
- name: Susanne Gerber
  affiliations:
  - 1
  - 10
  corresponding: true
  email: sugerber@uni-mainz.de
affiliations:
  1: Institute for Human Genetics, University Medical Center of the Johannes Gutenberg University Mainz, Mainz, Germany
  2: Institute of Molecular Biology (IMB) Mainz, Mainz, Germany
  3: Biocenter, Johannes Gutenberg University Mainz, Hanns-Dieter-Hüschi-Weg 17, 55128 Mainz, Germany
  4: Biocenter, International Max Planck Research School on Cellular Biophysics, Hanns-Dieter-Hüsch-Weg 17, 55128 Mainz, Germany
  5: Institute of Molecular Physiology, Johannes Gutenberg University Mainz, Mainz, Germany
  7: Institute of Pharmaceutical and Biomedical Sciences, Johannes Gutenberg University Mainz, 55128 Mainz, Germany
  8: Institut für Humangenetik und Genommedizin, Uniklinik RWTH Aachen, Aachen, Germany
  9: Institute for the History, Philosophy, and Ethics of Medicine, Johannes Gutenberg University Medical Center Mainz, Mainz,
    Germany
  10: Institute for Quantitative and Computer Biosciences (IQCB), Mainz, Germany
  11: Division of Epigenetics, DKFZ-ZMBH Alliance, German Cancer Research Center, 69120 Heidelberg, Germany
  12: Faculty of Biosciences, Heidelberg University, 69120 Heidelberg, Germany
  13: DKFZ Hector Cancer Institute at the University Medical Center Mannheim, 69120 Heidelberg, Germany
abstract: Direct RNA sequencing (DRS) is a nanopore-based technique for analyzing RNA in its native form, promising breakthroughs
  in diagnostics and biomarker development. Coupled to RNA002 sequencing chemistry, its clinical implementation has been challenging
  due to low throughput, low accuracy, and lack of large-scale RNA-modification models. In this study, we evaluate the improvements
  achieved by pairing the latest RNA004 chemistry with novel modified-base-calling models for pseudouridine and N6-methyladenosine
  using diverse RNA samples from cell lines, synthetic oligos, and human blood. Finally, we present the first clinical application
  of DRS by confirming the loss of RNA methylation in a patient carrying truncating mutations in the methyltransferase METTL5.
  Conclusively, the combined use of RNA004 chemistry with the base-calling models significantly improved the throughput, accuracy,
  and site-specific detection of modifications. From this perspective, we offer an outlook on the potential suitability of
  DRS for use in routine diagnostics and quality assessments of RNA therapeutics.
keywords: null
paper_type: research-article
subject_areas:
- Genomics
- Transcriptomics
- Medical Diagnostics
- Biotechnology
datasets:
- ClinVar
- Gencode
- Sequencing Quality Consortium (SEQC)
stats:
  word_count: 11994
  has_math: true
  section_count: 44
---
# Direct RNA sequencing enables improved transcriptome assessment and tracking of RNA modifications for medical applications

# Direct RNA sequencing enables improved transcriptome assessment and tracking of RNA modifications for medical applications

Charlotte Hewel$^{1*}$, Anna Wierczeiko$^{1*}$, Johannes Miedema$^{1}$, Felix Hofmann$^{1}$, Stephan Weißbach$^{1}$, Vincent Dietrich$^{1}$, Johannes Friedrich$^{1}$, Laura Holthöfer$^{1}$, Verena Haug$^{1}$, Stefan Mündnich$^{7}$, Lukas Schartei$^{3,4}$, Kristi Jenson$^{2}$, Stefan Diederich$^{1}$, Stanislav Sys$^{1}$, Tamer Butto$^{7}$, Norbert W Paul$^{9}$, Jonas Koch$^{11,12}$, Frank Lyko$^{11,13}$, Florian Kraft$^{8}$, Susann Schweiger$^{1}$, Edward A Lemke$^{3,5}$, Mark Helm$^{7}$, Matthias Linke$^{1,\S}$, Susanne Gerber$^{1,10\,\S,\ddagger}$  

$^{1}$Institute for Human Genetics, University Medical Center of the Johannes Gutenberg University Mainz, Mainz, Germany  

$^{2}$Institute of Molecular Biology (IMB) Mainz, Mainz, Germany  

$^{3}$Biocenter, Johannes Gutenberg University Mainz, Hanns-Dieter-Hüschi-Weg 17, 55128 Mainz, Germany  

$^{4}$Biocenter, International Max Planck Research School on Cellular Biophysics, Hanns-Dieter-Hüsch-Weg 17, 55128 Mainz, Germany  

$^{5}$Institute of Molecular Physiology, Johannes Gutenberg University Mainz, Mainz, Germany  

$^{7}$Institute of Pharmaceutical and Biomedical Sciences, Johannes Gutenberg University Mainz, 55128 Mainz, Germany  

$^{8}$Institut für Humangenetik und Genommedizin, Uniklinik RWTH Aachen, Aachen, Germany  

$^{9}$Institute for the History, Philosophy, and Ethics of Medicine, Johannes Gutenberg University Medical Center Mainz, Mainz, Germany  

$^{10}$ Institute for Quantitative and Computer Biosciences (IQCB), Mainz, Germany  

$^{11}$Division of Epigenetics, DKFZ-ZMBH Alliance, German Cancer Research Center, 69120 Heidelberg, Germany  

$^{12}$Faculty of Biosciences, Heidelberg University, 69120 Heidelberg, Germany  

$^{13}$ DKFZ Hector Cancer Institute at the University Medical Center Mannheim, 69120 Heidelberg, Germany  

$^{\dagger}$To whom correspondence should be addressed (sugerber@uni-mainz.de)  

Joint senior authors  

*These authors contributed equally.  

# Abstract  

Direct RNA sequencing (DRS) is a nanopore-based technique for analyzing RNA in its native form, promising breakthroughs in diagnostics and biomarker development. Coupled to RNA002 sequencing chemistry, its clinical implementation has been challenging due to low throughput, low accuracy, and lack of large-scale RNA-modification models. In this study, we evaluate the improvements achieved by pairing the latest RNA004 chemistry with novel modified-base-calling models for pseudouridine and $N^{6}$-methyladenosine using diverse RNA samples from cell lines, synthetic oligos, and human blood. Finally, we present the first clinical application of DRS by confirming the loss of RNA methylation in a patient carrying truncating mutations in the methyltransferase METTL5. Conclusively, the combined use of RNA004 chemistry with the base-calling models significantly improved the throughput, accuracy, and site-specific detection of modifications. From this perspective, we offer an outlook on the potential suitability of DRS for use in routine diagnostics and quality assessments of RNA therapeutics.  

# Introduction  

Naturally occurring modifications to RNA such as $N^{6}$-methyladenosine (m$^{6}$A) and pseudouridine ($\Psi$) crucially affect its structure, stability, and its interactions with proteins, and as such dynamically regulate molecular processes in cells. More than 170 chemical RNA modifications are currently known, and more are expected to be discovered [1].  

Modifications on mRNA molecules appear to be involved in translating, splicing, and stabilizing RNA [2], [3]. For example, pseudouridylation at stop codons can enable readthrough, allowing protein synthesis despite a "translation to stop" signal [4]. This mechanism has attracted significant attention in drug development, as approximately 10–20% of genetic mutations reported in the variant database ClinVar contain premature termination codons (PTCs). PTCs give rise to truncated proteins that cannot function as intended, leading to various inherited diseases. Translational readthrough-inducing drugs (TRIDs) show promise as therapeutic agents for a number of rare diseases [5]. Recent advances in this field include work by Scharf et al., who developed a model organelle system using DKC1 and small nucleolar RNAs (snoRNAs) as guide RNAs to achieve precise, site-specific pseudouridylation, enabling controlled translational readthrough at targeted transcripts [6].

Several aberrations of RNA-modifying enzymes are linked to human diseases, so called "modopathies" [7], [8], [9]. For example, loss of pseudouridine synthase PUS1 is associated with mitochondrial myopathy with lactic acidosis and sideroblastic anemia (MLASA), whereas dysfunctional PUS3 and PUS7 are associated with intellectual disability and neurodevelopmental delay [10], [11], [12]. Additionally, patients with dyskeratosis congenita have reduced pseudouridylation of 28S rRNA or the telomerase RNA component (TERC) [13], [14].

M$^{6}$A plays an important role in multiple cancers [15]. For example, the methyltransferase METTL3 can be upregulated in glioblastoma, thereby upregulating the expression of the target cancer gene SOX2. On a related note, epitranscriptomic rRNA fingerprinting approaches, to distinguish between tumor and normal samples from a fraction of reads -or ultra low depth sequencing- show promising avenues to classify cancers based on their epitranscriptomic signature at either a fraction of the cost or a fraction of the time classical approaches would take[16].  

Interestingly, the methyltransferase METTL5 is known to be responsible for the methylation of a specific adenine at position 1832 of the 18S rRNA, and is therefore essential for the translation process [17]. Dysfunction of METTL5 due to genetic mutations causes an intellectual developmental syndrome with severe microcephaly [18].

Consequently, both human diagnostics and research applications would benefit from a streamlined and high-throughput method for measuring RNA modifications.

Conventional RNA sequencing (RNA-seq) by next-generation sequencing (NGS) facilitates differential expression analysis of genes or transcripts and analysis of differential splicing in a high-throughput mode [19]. However, the required conversion from RNA into cDNA and the subsequent fragmentation erases a sizable quantity of information present on native RNA, such as modifications. Thus, conventional RNA-seq is not suited to directly observe either full-length isoforms or RNA modifications [20]. For clinical and research applications this indicates that metrics for RNA modifications could only be provided by indirect measurements, such as differences in the level of methyltransferase expression or splice isoforms.

Direct RNA sequencing (DRS) by Oxford Nanopore Technologies (ONT) is a major innovation, as it enables the detection of nucleotide-specific modifications directly on native RNA molecules by measuring real-time variations in electrical current [21]. DRS can quantify gene expression while simultaneously capturing full-length transcripts, splicing patterns, poly(A) tail length, and distinct RNA modifications within a single assay (Figure 1).

This capability for comprehensive profiling holds promise for advanced diagnostic and research applications, including enhanced detection of modopathies, accelerated development and quality control of mRNA therapeutics, and simplified epitranscriptomics analyses [22], [23].  

Although these prospects are promising, it is important to acknowledge the evolution of DRS technology. The chemistry of the now discontinued SQK-RNA002 sequencing kit for DRS was applied in various contexts, such as tRNA sequencing [24]. However, its performance was mixed owing to its low throughput, low accuracy, and the absence of modified-base calling within ONT basecallers [25], [26]. To address these limitations, ONT introduced the SQK-RNA004 sequencing kit, featuring updated flow cells, a new motor protein, and base-calling models capable of detecting the RNA modifications m$^{6}$A, Ψ and canonical nucleotides.  

In this study, we comprehensively compare the RNA002 and RNA004 chemistries using diverse RNA samples including cell cultures and human blood, evaluating improvements in yield, quality, gene coverage, poly(A) tail length estimation, and the detection of m$^{6}$A and Ψ modifications. To illustrate the practical implications of using RNA004 chemistry, we highlight two practical applications of DRS, a model system for future RNA therapeutics and the diagnostic of a rare disease case.  

First, we present an example for the performance of site-specific Ψ detection in RNA therapeutics by validating the expected stoichiometry introduced by a pseudouridylation system developed by Schartel et al. 2024, using both the RNA002 and RNA004 flow cells.  

Second, we showcase the first clinical application of direct RNA sequencing using RNA004 chemistry. Here, we confirmed the loss of function of the m$^{6}$A methyltransferase METTL5 in a patient harboring two compound heterozygous variants predicted to disrupt enzyme activity,  

classified as pathogenic and of uncertain significance (VUS). This highlights how DRS can improve the interpretation of VUS in RNA-modifying enzymes and be a promising tool for clinical diagnostics.  

# Material and Methods  

# Sample description  

Five different sample sources were used during this study and sequenced a total of 21 times. Universal Human Reference RNA (UHRR) was purchased from Thermo Fisher Scientific (cat. no. QS0639). The HEK293T cells were transfected with EGFP and mCherry. The human samples were taken from healthy volunteers or a patient after written informed consent was obtained.  

# HEK293T samples  

HEK293T cells were transfected with artificial snoRNAs as well as EGFP and mCherry. The snoRNAs were designed to target a premature stop codon within the EGFP and mCherry transcripts at nucleotide positions 115 and 565, respectively. In Case A, both mRNAs were targeted for pseudouridinylation; in Case B, the mCherry mRNA was preferentially targeted owing to a decrease in EGFP pseudouridinylation. In the Control condition, a scrambled snoRNA was transfected, as were mCherry and EGFP [6].  

# Direct RNA library preparation for the cell line samples and RNA002/RNA004 chemistries  

For direct RNA library preparation, we used either the old DRS chemistry (SQK-RNA002, ONT) or the updated kit (SQK-RNA004, ONT) following the manufacturer's protocol. In brief, 100 ng of poly(A)-tailed RNA or 1000 ng of total RNA was adjusted to 9 μl with nuclease-free water. To this RNA sample, 3 μl of NEBNext Quick Ligation Reaction Buffer (New England Biolabs, B6058), 1 μl RT Adapter (RTA, ONT), and 1.5 μl T4 DNA Ligase (2×10^6 U/ml; New England Biolabs, M0202) were added, resulting in a total volume of 14,5 μl. The reaction was mixed by pipetting and incubated for 10 min at room temperature. Next, the reverse transcription master mix was prepared by mixing 9 μl of nuclease-free water, 2 μl of 10 mM dNTPs, 8 μl of 5× first-strand buffer (Thermo Fisher Scientific), and 4 μl of 0.1 M DTT. This master mix was added to the RNA sample containing the RT Adapter-ligated RNA along with 2 μl of SuperScript III reverse transcriptase. The reaction was incubated at 50°C for 50 min then at 70°C for 10 min, and then cooled to 4°C. RNAClean XP beads (72 μl; Beckman Coulter, A63987) were then added to the reaction, followed by incubation on a Hula mixer for 5 min at room temperature. Subsequently, the sample was washed twice with 70% ethanol, and the DNA was eluted with 20 μl of nuclease-free water. The eluted DNA was used in the adapter ligation reaction. For that reaction, 8 μl of NEBNext Quick Ligation Reaction Buffer, 6 μl of RNA Adapter (RMX for RNA002; RLA for RNA004), 3 μl of nuclease-free water, and 3 μl of T4 DNA Ligase were mixed with the 20 μl of eluted DNA (total volume: 40 μl). The reaction was incubated for 10 min at room temperature. After incubation, 20 μl of RNAClean XP beads were added to the adapter ligation reaction, followed by incubation on a Hula mixer for 5 min at room temperature. The sample was then washed twice with Wash Buffer (WSB, ONT) using a magnetic rack. Next, the pellet was resuspended in 41 μl (RNA002) or 33 μl (RNA004) of Elution Buffer (EB) and incubated at 37°C for 10 min in a Hula mixer to release long fragments from the beads. Finally, the eluate was cleared by pelleting the beads on a magnet, retained, and transferred to a clean 1.5 ml tube. One microliter of reverse-transcribed and adapted RNA was quantified using a Qubit fluorometer. For R9.4.1 PromethION sequencing (RNA002), 40 μl of the library was mixed with 35 μl of nuclease-free water and 75 μl of RRB and loaded into a R9.4.1 PromethION flow cell. For PromethION sequencing (RNA004), 32 μl of library was mixed with 100 μl of Sequencing Buffer (SB) and 68 μl of Library Solution (LIS) and loaded into an RNA chemistry PromethION flow cell. For the 18S rRNA sample, a MinION RNA flow cell (FLO-MIN004RA) was loaded in accordance with the manufacturer's instructions.  

# Peripheral blood and in vitro transcription  

The peripheral blood was obtained from a healthy volunteer. The RNA was extracted using the PAXgene Blood miRNA Kit from Qiagen according to the manufacturer's protocol, except the RNA was eluted in nuclease-free water instead of the buffer provided. The RNA was characterized using the Bioanalyzer total RNA Nano Assay according to the manufacturer's protocol. The RNA had a concentration of ng/μl and a RIN of 7.2. Depletion of globin mRNA was performed with the GLOBINclear-Human Kit from ThermoFisher Scientific (AM1980) according to the manufacturer's protocol; this was carried out four times. The total input of RNA was 20 μg, the total output was 11 μg of globin-depleted RNA. The concentration was measured using the Qubit RNA HS Assay from ThermoFisher Scientific. Two micrograms of RNA was stored for later use in the direct RNA Run. Nine micrograms of RNA was taken forward to the poly(A) selection using an NEBNext Poly(A) mRNA Magnetic Isolation Module according to the manufacturer's protocol. The poly(A) enrichment was carried out three times; the total output was 23 ng of mRNA measured with Bioanalyzer. The sample had an average length of \~1kb. The sample concentration was measured again using the aforementioned Qubit assay. The subsequent reverse transcription (RT), PCR, IVT, polyadenylation, and 5' capping were carried out according to Tavakoli et al. (2023). The following individual amendments were made: the IVT primers used in the PCR had a final concentration in the reaction of 0.5 μM per primer. The input amount of mRNA used for the RT and PCR was 7.1 ng; the output was 905 ng of cDNA, measured with the Qubit DNA HS Assay (Thermo Fisher Scientific). The IVT was carried out twice, each with an input of 126.7 ng cDNA. The output was pooled to give a final amount of 4.9 μg RNA, as measured with the Qubit RNA HS assay. Libraries were prepared using the SQK-RNA004 sequencing kit (ONT). The library output was 167 ng of RNA/cDNA hybrid, as measured with the Qubit DNA HS Assay. The library was loaded completely onto the PromethION RNA Flow Cell (FLO-PRO004RA).  

# Base calling and alignment of RNA002 and RNA004 runs  

The raw pod5 files from all RNA004 sequencing runs were base-called using Dorado v0.7.2 with the canonical base-call model rna004_130bps_sup@v5.0. The model allowed for direct calling of m$^{6}$A and Ψ using the flag --modified-bases m6A pseU. Poly(A) tail lengths were also estimated by including flag --estimate-poly-a, after the tailfindr algorithm that was recently adopted by ONT. The base calling of raw pod5 files from the RNA002 sequencing runs was done with Dorado's high accuracy model for RNA002, that is, rna002_70bps_hac@v3. Base-called reads of all samples were then aligned to the primary assembly of the human reference genome hg38, downloaded from Gencode release 43 (https://ftp.ebi.ac.uk/pub/data-bases/gencode/Gencode_human_release_43/GRCh38.primary_assembly.genome.fa.gz). Alignment was performed in Minimap2 v2.26 with the following settings: -y --MD -ax splice -uf -k14. The resulting BAM files were sorted and indexed using samtools v1.16.1. The HEK293T samples were additionally mapped onto the EGFP and mCherry reference sequences for analyzing the detection of modified targets. The oligos were mapped in addition to their custom oligo references (see Schartel et al. 2024). The quality metrics of all sequencing runs and mappings were derived by NanoComp v1.23.1. The average base-call quality, the alignment-based percent identity and the N50 read length were visualized in Python 3.8 using matplotlib v.3.8.3 and seaborn v.0.13.2. The percentage mismatch on chromosome 20 for the cell line samples was performed using dRNA-eval, after realignment as described on GitHub (https://github.com/KleistLab/nanopore_dRNAseq) and subsequently plotted in R.  

# Modification information extraction  

The modification bed files were generated from the Dorado base-called modbam files with modkit version 0.3.1. For m$^{6}$A the reads were subset to DRACH regions with the flag --motif DRACH 2, additionally the flags --ignore 17802 and --filter-threshold A:0.8 --mod-threshold m:0.98 were used, as determined by the modification probability histogram also made with modkit (see Figure S3). For Ψ the flags --ignore a, --filter-threshold T:0.8, and --mod-threshold m:0.98 were used. Then the bedfiles were filtered to have a valid coverage of at least 20 reads and a site-specific methylation of at least 5% to reduce false positives.  

# Mendeliome counts  

featureCounts v2.0.0 tables for genes > 0 or genes > 10 reads coverage were intersected with an in-house list of genes associated with known diseases (mendeliome) using bedtools v2.27.1. Plots were generated in R using ggplot2 v3.4.4.  

# Illumina control data  

Comparative data from Illumina were obtained from the Sequencing Quality Consortium (SEQC) [29]. We downloaded the Universal Human Reference RNA (UHRR) Illumina HiSeq 2000 subset of the study and processed the data according to best practices (GSE47774).  

# Annotation of genomic features  

Reads were mapped to genomic features using featureCounts v.2.0.6. Basic gene annotation downloaded from Gencode v43 served as the annotation reference (https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_43/gen-code.v43基本.annotations.gtf.gz). Parameter -L was passed to featureCounts to account for long reads as input; -s 0 to perform unstranded read counting. The format of the annotation file was specified with -F 'GTF'.  

# Analysis of Ψ and m$^{6}$A detection at target positions  

For the site-specific analysis of modification, the Dorado-derived modification probabilities as well as the mismatch frequencies were extracted from the ML/MM tags of the respective bam files using pysam v0.22.1 with $min\_base\_quality=13$ and $threshold=0.8$ (Python 3.8, pysam: https://github.com/pysam-developers/pysam). Additionally, we performed Dorado-based $\Psi$ calling on reads harboring misbasecalled Cs by changing the motif specification to $motif="C"$ in the conFiguretoml file of the respective Dorado base-calling model.  

For the EGFP and mCherry motifs as well as the known Ψ-site on the PSMB2 transcript, the frequencies of U-based Ψs, C-based Ψs, unmodified Cs and unmodified Us were calculated and plotted in R v4.2.2 using the R package ggplot2 v3.4.4.  

To extract the m$^{6}$A modification frequencies for all 18S rRNA transcripts, we additionally mapped the raw reads onto the rDNA reference sequence published by George et al. [67]for the peripheral blood samples from two healthy individuals and one patient, as well as the 18S rRNA IVT sample. The m$^{6}$A modification probabilities at the 18S rRNA position A1832 for the peripheral blood and the 18S rRNA IVT samples were extracted using pysam with min_base_quality=13 and threshold=0.8 and plotted using the Python package seaborn v0.13.2.  

# Estimation of poly(A) tail length  

Poly(A) features were extracted during base calling with Dorado by adding the flag –estimate-polya as detailed above. For the basic comparison between tailfindr and Dorado 0.7.2 a test data set for RNA002 chemistry was downloaded from ERR3349888. The raw single fast5 data was subsequently transferred into multi fast5 via single_to_multi_fast5 from ont-fast5-api toolkit and transferred into pod5 via pod5 convert fast5. Then base calling was performed with Dorado 0.7.2 rna002_70bps_hac@v3 model and --estimate-poly-a flag. The poly(A) length was extracted from the resulting ubam file by storing the pt tag of each read in a table. For the tailfindr length estimation, we made use of the information on pre-existing length and the bar-coding table as provided by the analysis of Krause and Niazi (https://github.com/adnani-azi/krauseNiazi2019Analyses). Subsequently, the data was loaded in Jupyter Notebook and plotted with seaborn. The first 200,000 reads of the IVT blood data were extracted from the RNA002 and RNA004 samples, and for the genes DDX17, OLA1 and SRP14, the bam file was filtered with samtools and then the pt tag was stored in a table and plotted in seaborn. For the  

transcriptome sample distribution, unique reads aligned to GRCh38 were retained and plotted in R with ggplot2.

# m$^{6}$A calling using mAFiA and m6ABasecaller for chromosome 20  

The data was subset to chromosome 20 via filtering by samtools. Then, pod5 filter was used on the read IDs to retain a subset of the raw data for chr20. pod5 convert to_fast5 was used to transfer data into fast5 as required for downstream analysis with the base callers for m$^{6}$A RNA002. Both mAFiA and m6ABasecaller were run with default options as described in (https://github.com/dieterich-lab/mAFiA) & (https://github.com/novaalab/m6ABasecaller). The GLORI test data set was obtained from Liu et al. (2022). Dorado 0.7.2 and modkit were run as described previously. Plotting was done in Python using UpSetPlot version 0.9.0.  

# RNA isolation and preparation for GLORI and direct RNA control sequencing  

Total RNA from 3 biological HEK293T replicates was isolated using TRIzol. Small RNA species were depleted using the MEGAclear Transcription Clean-Up Kit (Thermo Fisher Scientific). mRNA enrichment was performed twice using the Dynabeads mRNA Purification Kit (Thermo Fisher Scientific).  

For direct RNA sequencing, 300 ng of mRNA pooled from the three biological replicates were sequenced on a single flow cell on the MinION Mk1B platform using the direct RNA sequencing kit (SQK-RNA004; Oxford Nanopore). Data analysis was performed as detailed above.  

For GLORI sequencing, the mRNA was fragmented at 94 °C for 3 min using the NEBNext Magnesium RNA Fragmentation Module (New England Biolabs) and purified using the RNA Clean & Concentrator-5 kit (Zymo Research) including a DNase I-digestion step. mRNA protection, deamination, and deprotection were performed as described in literature [40], [41]. For preparing the sequencing libraries, RNA samples were end-repaired via Antarctic phosphatase (New England Biolabs) and T4 Polynucleotide Kinase (New England Biolabs) treatments according to the manufacturer's instructions. End-repaired samples were purified using the RNA Clean & Concentrator-5 kit (Zymo Research). Sequencing libraries were then prepared using the NEBNext Small RNA Library Prep Set for Illumina in combination with the NEBNext Multiplex Oligos for Illumina (Index Primer Sets 1 and 3) (New England Biolabs). Sequencing was performed by the Next Generation Sequencing Core Facility of the German Cancer Research Center, Heidelberg on a NovaSeq 6000 platform (Illumina) using a 100 bp paired-end sequencing protocol. Sequencing adaptors from raw reads were removed by Trim Galore (version 0.6.6). Trimmed reads were further processed by the GLORI-tools pipeline. GLORI-tools is available on GitHub: https://github.com/liucongca/GLORI-tools. Software used for executing the GLORI-tools pipeline included python (version 3.10.1), samtools (version 1.19), STAR (version 2.7.10a), and bowtie (version 1.3.0). The human genome (GRCh38) and transcriptome (GCF_000001405.39) reference files were obtained from UCSC. To investigate the correlation of methylation ratios between DRS and GLORI-seq samples of HEK293T cells, replicates were merged by averaging the methylation ratios across overlapping m6A sites. Bivariate density plots were generated using ggplot version 3.5.1, the goodness-of-fit measure R^2 was calculated using base R version 4.3.2 to assess the correlation between methylation ratios.  

# Base-calling error pattern extraction and pseudouridine calling using NanoCEM in reporter sequences EGFP and mCherry  

nanoCEM version 0.0.6.1 was run with default options for the HEK293T samples A, B, and C, and positions 115 and for the sequences of EGFP and mCherry.  

# U-C mismatch analysis on high-confidence pseudouridine sites  

Perbase version 0.9.0 base-depth was run on the high-confidence sites from Tavakoli et al. The mismatches according to the reference (GRCh38) were extracted from the resulting table and plotted in R using ggplot2.  

# 18S rRNA Methylation Control Sample Plasmid preparation and in vitro transcription  

The target sequence was cloned into a pUC57 vector, which included an internal T7 promoter, the desired template sequence, and a BshTI restriction enzyme site at the 3' end. Linearization of the plasmid was carried out overnight, following manufacturer's instructions (Thermo Fisher Scientific). Next, the plasmids were purified using phenol–chloroform extraction followed by ethanol precipitation. Successful linearization and the quality of the plasmids were confirmed by agarose gel electrophoresis and analysis with a NanoDrop One spectrophotometer.  

IVT was carried out using the HiScribe T7 High Yield RNA Synthesis Kit (New England Biolabs) according to the manufacturer's instructions. In brief, 2 µg of linearized plasmid was used as the template, along with 10× Reaction Buffer, 10 mM NTPs, and 2 U of T7 RNA Polymerase Mix. The reaction mixture was incubated at 37°C for 2 hours, and the process was stopped by digesting the template plasmid with DNase I (Thermo Fisher Scientific, EN0525) according to the manufacturer's protocol. The resulting RNA was purified using the Monarch RNA Cleanup Kit (New England Biolabs, T2040), and the quality of the product was evaluated by capillary electrophoresis using Agilent RNA ScreenTape Analysis.  

# Patient sample and data processing  

Genomic DNA was isolated from the patient's blood sample. Subsequently, all coding exons including flanking intron sequences of genes were enriched ("target enrichment" by hybridization) up to positions +/-20 using the SureSelect QXT Exome V7 Enrichment System (Agilent). The 2×150 bp (paired-end) NGS was performed on the NextSeq 500 System (Illumina) using the NextSeq 500/550 High-Output v2 Kit (300 cycles) reagents (Illumina).  

The sequenced Illumina data were first converted to fastq files using bc12fastq v2.20.0.422 and subsequently mapped onto the human reference genome hg19 using the bwa-mem aligner integrated in the Clara Parabricks Workflows (pbrun fq2bam) from NVIDIA (version 4.0.0-1).  

# METTL5 RT-PCR assay  

To determine splice aberration in the patient sample, a METTL5-gene-specific PCR was performed. The RNA was reverse-transcribed into cDNA using the PrimeScript RT Reagent (Takara) according to the manufacturer's protocol. cDNA was amplified using a METTL5 gene-specific PCR, targeting exons 1–7. Primer sequences are provided in Table S6. The FastStart High Fidelity PCR System (Roche) was used according to the manufacturer's protocol, except that only a 25 μl reaction was prepared. The annealing temperature was 60°C. The elongation time was 2 min, with 35 cycles in total. The product was quantified using the Qubit DNA BR Assay.  

Library preparation was performed using the Ligation Sequencing Kit (SQK-LSK114, ONT). The library was loaded completely into a PromethION DNA Flow Cell (FLO-PRO114M) and sequenced for approximately 9 hours.  

The data was aligned against GRCh38 and only reads mapping to METTL5 were retained by filtering with samtools. Subsequent plotting was done with ggsashimi v1.1.5.  

# Results  

# Enhanced performance and yield of RNA004 chemistry in direct RNA sequencing  

A total of 21 flow cells using either RNA002- or RNA004-compatible settings were sequenced on a PromethION sequencer and two test samples on a MinION (Figure 2A, Table S1). First, we compared the performance of the two chemistries in terms of throughput and quality. To this end, we sequenced three different types of samples on a PromethION device using both  

RNA002 and RNA004 chemistry: 1) Universal Human Reference RNA (UHRR; total RNA), 2) poly-selected RNA from HEK293T cells, and 3) total RNA from human peripheral blood from a healthy individual (Figure 2A). All runs were base-called using the recently released Dorado base-caller (version 0.7.2, ONT). UHRR was used as a technical control to ensure comparability with previously published data from ONT. The HEK293T cells were additionally transfected with a newly developed pseudouridylation system and two mRNA reporter sequences (EGFP and mCherry) carrying Ψ-target sites, which were utilized later in the study to evaluate the detection of RNA modifications (Figure 2A, Table S1) [6]. The RNA obtained from human blood was sequenced both in a native form and after in vitro transcription (IVT). IVT incorporates unmodified RNA nucleotides only and erases natural poly(A) tails from the native molecules and is therefore used as an unmodified control for the purpose of detecting modifications (see Material and Methods, Figure 2A, Table S1).

Following sequencing, we determined that the overall yield is dependent on the chemistry type (RNA002 or RNA004), method of library preparation (poly(A)-selected or total RNA), and sample origin (standardized cell line or peripheral blood; Figure 2B,C; Table S1). For all samples of the same composition, the RNA004 chemistry delivered higher yields than RNA002 (Figure 2B,C; Table S1).  

The IVT samples, using either chemistry, showed the lowest throughput with < 2 gigabases (Gb) and < 3 million reads. The highest yield was observed for poly(A)-selected RNA from HEK293T cells using RNA004 chemistry, with a yield of approximately 17.3 and 21.94 Gb and more than 18 million reads. By contrast, the same samples sequenced with SQK-RNA002 yielded less than 35% of the throughput observed with RNA004 and < 7 million reads (Table S1).  

Consistent with a general increase in throughput, the RNA004 runs show an increased average base quality derived from the Phred-based Q-scores, as well as better percent reference identity than the RNA002 chemistry, with mean scores close to 98% (Figure 2D,E; Table S1). For both chemistries, the most frequent source of base-calling errors were insertions and deletions, which is a common bias for ONT data (Figure 2F) [27], [28].  

# Enhanced gene expression profiling with RNA004  

We further analysed the transcriptional patterns, including gene expression to assess the utility of RNA004 chemistry in clinical settings. The number of genes covered with at least 10 reads (10X) was greater than if RNA002 chemistry was used (Figure 3A). This observation correlates with the higher throughput observed for the RNA004 chemistry (Table S1, Table S2). To check whether a greater sequencing depth would increase the number of genes detected, we calculated the combined number of 10X-covered genes of all samples for RNA002 and RNA004 to be five and seven samples, respectively. Then we compared the combined number of distinct detected genes per chemistry with that of ultra-deep sequenced NGS samples by extracting the gene counts from 17 publicly available UHRR datasets from the SEQQC study [29]. The sum of all RNA004 runs were able to capture a comparable number of distinct annotated genes to that of NGS-derived results, while the sum of RNA002 data detected fewer features (Figure 3A).  

Of particular interest for clinical applications is the coverage of genes associated with Mendelian disorders (the Mendeliome). In our analysis, the proportion of distinct disease-associated genes with at least 10X coverage ranged from 29% up to 75%, with variation depending on the sample type (Figure 3B, Table S2, Table S3). The native human blood sample, as example of a routine diagnostic tissue, covered nearly 60% of disease-associated genes when sequenced with RNA004 chemistry.  

Furthermore, we examined the gene body coverage from the 5' to 3' end and observed similar coverage across all samples sequenced using either chemistry (Figure S1).  

# Evaluation of estimates of native poly(A) tail length  

Altered polyadenylation patterns on mRNA molecules are linked to various diseases, including cancer and neurological disorders, and can serve as diagnostic biomarkers or as potential therapeutic targets through their effects on gene regulation and protein expression [30], [31].  

DRS can detect the length of a poly(A) tail natively via the length and duration of the raw signal pattern for poly(A). Tailfindr was one of the first tools developed for estimating poly(A) tail length from DRS data sequenced with RNA002 chemistry [32]. Recently, ONT integrated such functionality into their production base-caller Dorado for both RNA002- and RNA004-derived data. As part of our study, we evaluated the estimation of poly(A) tail length by ONT's base-caller Dorado in comparison to tailfindr and examined the difference in quantity of polyadenylation between the old and new chemistry in our samples.  

A test data set with known poly(A) tail lengths (10–150 bp) was obtained from the tailfindr publication and re-base-called with poly(A) tail estimation using Dorado version 0.7.2. Dorado revealed similar results to both the original tailfindr assessment and the empirical label (Figure 3C).  

In our samples, the estimation of poly(A) tail length by Dorado revealed similar distributions across all samples and chemistries (Figure 3E,F; Figure S2). For the IVT samples derived from blood, in which natural poly(A) tails are replaced by artificially added adenosines due to reverse transcription and in vitro polyadenylation, the expected Gaussian distribution of similar poly(A) tail lengths across all transcripts was confirmed and no difference between RNA002 and RNA004 was observed (Figure 3D).  

Moreover, we examined the estimation of poly(A) tail length for the genes $DDX17$, $SRP14$, and $OLA1$ with long, middle, and short poly(A)-tailed transcripts, respectively. A comparison between RNA002 and RNA004 shows high concordance for all three polyadenylation patterns (Figure 3 G–I).  

# Modifications detected in RNA004 samples reveals multiple uniquely modified sites  

With the release of the RNA004 chemistry, the functionality for detecting m$^{6}$A modifications in DRACH motifs and transcriptome-wide Ψ was integrated into Dorado. ONT discontinued the RNA002 kit in March 2024, which explains the absence of a production modification calling model for this version. Nevertheless, third-party tools for RNA modification detection using RNA002 have been developed and considerably advanced the field of epitranscriptomics in the past [33], [34], [35], [36], [37].  

We first compared the performance in detecting m$^{6}$A in RNA002 and RNA004 samples. For the RNA004 samples, we used Dorado-based m$^{6}$A calling, whereas for those sequenced with the older RNA002 chemistry we utilized two community-developed m$^{6}$A-detection tools: mAFiA and m6ABasecaller (see Materials and Methods, Figure S5) [38], [39].  

Using RNA004 samples in combination with the ONT base caller detected the highest number of m$^{6}$A sites, whereas applying m6ABasecaller and mAFiA to RNA002 samples resulted in significantly fewer m$^{6}$A sites; measurements were made for chromosome 20 (Figure 4A).  

Furthermore, we compared our results to a reference m$^{6}$A set from the literature generated by GLORI, an NGS-based sequencing method for transcriptome-wide quantification of m$^{6}$A [40]. When examining the transcriptome-wide distribution of methylated DRACH motifs using RNA004-sequenced HEK293T samples, the number of distinct m$^{6}$A predictions made by  

Dorado was much higher than the intersection with the GLORI reference set, as observed for chromosome 20 (Figure 4B).  

# Transcriptome-wide cross-correlation of GLORI m6A with DRS HEK293T data  

Next, GLORI sequencing was repeated in-house for three replicates of HEK293T cells (see Figure 4 A,B & Figure S4). Overall, RNA004 chemistry on the P24 in connection with the new Dorado model recovered the most m6A sites, followed by the GLORI set by Liu et al., followed by the in-house GLORI data (see Figure S4A). The RNA002 data set in combination with the legacy basecallers recovered the least m6A sites (Fig 4 A, B). Apart from the sequencing depth and higher cut-offs by the legacy basecallers (mAFiA has a sequencing-depth cut off of 50 reads for example), it should be mentioned at this point that the in-house data set consisted of three replicates, the other data sets had two replicates, meaning that a more stringent overlap was preferred to a higher number of total sites to be reported. The cross correlation of sites was highest between the two replicates of HEK293T generated by the P24, with a value of R² of 0.94 (Figure S4B). The GLORI set from Liu et al., showed a correlation of R² = 0.88 with the in-house GLORI data and the Promethion RNA004 DRS showed a correlation of 0.86 with the in-house GLORI data and 0.85 with the GLORI reference data from literature (Figure S4 E,F,H).  

# Detection of modifications in a blood sample as an exemplary clinical tissue  

Next, we investigated the performance of m$^{6}$A and Ψ base calling by Dorado in the RNA004-sequenced native blood sample of a healthy individual compared to its corresponding unmodified IVT sample. More than 120,000 m$^{6}$A sites were predicted in the native blood sample. For the unmodified IVT sample, the number of probably false-positive predicted m$^{6}$A sites was 7,235, which is, however, only a minor fraction of the natively modified counterpart (Figure 4C).  

Furthermore, we were interested in the number of modification sites in genes associated to Mendelian disorders, the so-called "Mendeliome", which can be predicted by the Dorado m$^{6}$A caller. From all m$^{6}$A sites detected in a healthy blood sample, more than 35,000 were in Mendeliome genes (Figure 4D). Potential aberrations of modification stoichiometry in these regions might influence the function of gene products, their physiology and pathophysiology.  

By examining the average m$^{6}$A frequency across modified DRACH motifs, we observed that the natively modified blood and HEK293T samples have a characteristic distribution as reported in literature [37], [39], [41]. In contrast, the frequency of false-positive detected m$^{6}$A sites in the unmodified IVT sample was evenly distributed across the DRACH motifs (Figure 4H).  

The transcriptome-wide Ψ scan by Dorado predicted the existence of 600,000 potential modification sites with a modification frequency of at least 5% and a valid coverage of 20 reads, whereas approximately 20,000 false positives remained in the IVT sample (Figure 4E). Of the > 60,000 stop-codon sites in the human transcriptome, \~1% were predicted to be modified (Figure 4G). Pseudouridylation can trigger protein readthrough [4], making these predicted modification sites an interesting target for investigating premature termination codons (PTCs). Approximately 20,000 pseudouridylation sites were found in the Mendeliome (Figure 4F).  

Given the recent reports precedence for U>C mismatch at Ψ sites, we wondered how this would manifest in a transcriptome-wide manner [42]. Therefore, we queried the U>C mismatch at all high-confidence sites published by Tavakoli and co-workers for both RNA002 and RNA004 sequencing of the HEK293T samples. Using RNA002 chemistry, the average percentage U>C mismatch in reads was around 16%; RNA004 samples yielded an average of 6%, demonstrating a lower percentage of misbasecalls in the samples sequenced with RNA004 chemistry (Figure 4; Table S4).  

# RNA004 accurately reads the pseudouridylation stoichiometry in a targeted reporter system  

Next, we investigated detecting modifications in a site-specific manner, since another question is whether the technique is mature enough to track and target known positions in order to be developed into a clinical assay.  

First, validated the Ψ stoichiometry determined from a custom targeting pseudouridylation system developed by Schartel and co-workers [6]. Three HEK293T samples were transfected with a modified pseudouridine synthase (DKC1), artificial guide snoRNAs as well as a selectivity reporter sequence containing EGFP and mCherry sequences, each harboring a target motif that represents a premature stop codon. In sample A, both EGFP and mCherry motifs are expected to be equally targeted for pseudouridylation, whereas in sample B, the mCherry motif is preferentially targeted and therefore expected to be modified to a greater extent than EGFP (Figure 5A). Sample C contained a scrambled guide RNA with no pseudouridylation capability and is used as an unmodified biological control.  

First, we checked whether the targeted sites in the mCherry and EGFP mRNAs can be detected by both RNA002 and RNA004 chemistry by comparing the base-calling errors between targeted samples A and B to those of the non-pseudouridylated sample C using nanoCEM [43]. Both chemistries identify the modified sites in both reporters and samples based on a high U>C mismatch rate (Figure 5B–E). As shown on the transcriptome-wide level, the frequency of base-calling errors at canonical bases is lower for samples sequenced with RNA004 chemistry (Figure 5B–E).  

Next, we evaluated the performance of the Dorado-based Ψ detection for RNA004 based on 1) synthetic oligos that contain the EGFP and mCherry motifs of the targeted reporters both in fully modified and unmodified states; 2) the pseudouridylation-targeted sites of the EGFP and mCherry reporters in sample A and B; and 3) a high-confidence Ψ site in the PSMB2 transcript of HEK293T cells (Figure 5A, see Materials and Methods) [44].  

For both unmodified motifs in the synthetic oligos, the number of false-positive detected Ψ was rather low, with less than 2.5% of reads modified as determined by Dorado (Figure 5F, Table S5). The fully pseudouridylated mCherry motif revealed 60.69% modified reads, whereas for the EGFP site with the same expected Ψ stoichiometry, only 10.20% of reads were found to be modified (Figure 5F). Interestingly, the EGFP motif reveals a particularly high U>C mismatch rate (\~90% of reads) and as Dorado detects modifications at read level, considering only U-sites, C-called bases are neglected. When combining the number of C-mismatches with the number of modified reads called by Dorado both on U and C bases (Figure 5 B–G, Table S5; see Materials and Methods), the percentage of modified reads increases to 98.38% and 93.77% for the positive controls of the EGFP and mCherry motifs, respectively (Figure 5G).

The same pattern was observed for the targeted EGFP and mCherry reporters in samples A and B and only if both the C-mismatch and base-caller-derived Ψ sites were used can the expected differences in stoichiometry between samples A and B be verified [6]. Specifically, the modification frequency ratio between mCherry and EGFP was higher in sample B (9.6-fold) compared to sample A (2.6-fold).  

Moreover, the high-confidence Ψ site in PSMB2 transcripts reveals similar modification frequencies in both HEK293T samples, and by using both C-mismatches and Dorado-called Ψ sites, the modification frequencies were 11% above the expected stoichiometry of 80% modified reads reported in literature [44]. However, the number of Ψ sites discovered by Dorado amounted to only 40 and 42% of reads, respectively, which can be explained by Dorado's basecalling and modification model architecture.

Interestingly, for the fully modified EGFP motif of the oligo 1, Dorado predicts an m$^{6}$A modification directly next to the Ψ site (+1), which should not be present in the synthesized sequences (Figure S6). This shows that the performance of Dorado-based modification detection is dependent on the sequence context.

of accuracy, deletion, insertion, and mismatch rate are shown in relation to GRCh38 (dRNA-eval).

# Putative loss of function in METTL5 and site-specific m$^{6}$A detection

Finally, we present a clinical case from the Institute of Human Genetics Mainz for which we were able to validate the functional impact of genetic variations within a methyltransferase gene that is responsible for a site-specific m$^{6}$A modification using DRS.

A one-year-old girl showed severe microcephaly (occipito-frontal head circumference > -6 standard-deviation) and developmental delay. Two compound heterozygous variants, c.224+5G>A (p.(?)) and c.427A>T (p.(Lys143*)), in the METTL5 gene (NM_014168.4) were identified by whole exome sequencing and suspected as the underlying cause for an autosomal recessive intellectual developmental disorder type 72 (OMIM # 618665) (Figure 6A–C).

While the nonsense variant c.427A>T (p.(Lys143*)) in exon 4 of the METTL5 gene was classified as pathogenic based on ACMG guidelines (pathogenicity level 5, PVS1+PM2=8+2=10 points), the intronic variant c.224+5G>A was classified as a variant of unclear significance (VUS, pathogenicity level 3, PP3+PM2+PM3=1+2+2=5 points). To investigate the effect of the VUS on METTL5 splicing, the RNA extracted from peripheral blood of the patient was sequenced using DRS (RNA004) and revealed skipping of exon 2 in approximately 50% of the reads, suggesting a loss-of-function splicing defect. To validate this analysis and increase the vertical coverage of METTL5 transcripts, a targeted RT-PCR assay of exons 1–7 of the METTL5 transcript was applied and additionally confirmed the aberrant splicing pattern (Figure 2A and Figure 6D).

Since METTL5 is known to elicit an m$^{6}$A modification not globally but at a single site that is close to the active site of the ribosome (Figure 6E), we were interested whether we could verify aberrant m$^{6}$A modification in the peripheral blood of this patient. The patient sample showed reduced m$^{6}$A modification at the METTL5 target position 1832 of the 18S rRNA compared to healthy pediatric and adult samples (Figure 6F,G). For the first time, we can confirm the loss-of-function of an RNA modifying enzyme in a clinical case via DRS.

-  

# Discussion  

DRS has revolutionized RNA analysis by enabling the detection of both full-length transcripts and transcriptome-wide modifications from native RNA molecules. This approach has the potential to deepen our understanding of the complex epitranscriptome, which encompasses over 170 known modifications. The recently released RNA004 chemistry, featuring new base-calling models and integrated capabilities to detect modifications such as m$^{6}$A and Ψ offers exciting prospects for establishing DRS as a routine tool in both epitranscriptomic research and clinical applications.  

# RNA 002 versus RNA004  

In this study, we assessed the RNA004 chemistry compared to the earlier RNA002 version, focusing on improvements in sequencing quality, throughput, and novel capabilities for detecting modifications and demonstrated the first clinical application of DRS.  

In 2019, the first study to comprehensively analyze DRS for a human poly(A)-selected RNA derived from a cell line utilized 30 MinION flow cells to generate 9.9 million aligned reads with a median identity of 86% and a maximum read length of 21,000 bases [45]. This effort was shared between six different institutions.  

In our study, both single PromethION flow cells loaded with poly(A)-selected RNA from HEK293T cells provided each more output with higher quality at a fraction of the previous cost and effort.  

However, some limitations of the older chemistry, such as read lengths and transcriptome assessment, persist [26]. In particular, capturing full-length transcripts remains problematic, largely due to a mismatch between annotated transcript lengths and the fraction of reads covering the annotated 5' end. This issue arises from the DRS adapter design and is compounded by the fact that the motor protein, which translocates RNA through the pore, eventually releases the 5' end when the molecule is fully processed. Previous studies have shown that this results in the last few nucleotides being unsequenced, a challenge shared with the new chemistry as the adapter attaches only to one end of the RNA molecule [45]. This limitation also impacts state-of-the-art transcript-detection tools, such as bambu, which struggle to accurately predict the 5' end, even in "full-length" direct RNA reads [46]. Nevertheless, special strategies for preparing sequencing adapter libraries, combined with specific changes to MinKNOW's read detection algorithms, can partially mitigate the issue of incomplete or missing transcripts [24], [47]. For a comprehensive and up-to-date overview of isoform detection, we recommend the LRGASP study [48].  

# Site-specific modification stoichiometry  

The integration of models for detecting m$^{6}$A and Ψ modifications in Dorado has created opportunities to utilize DRS in routine analyses. By taking advantage of the new RNA004 chemistry and enhanced capabilities for detecting RNA modifications, we demonstrate precise estimation of site-specific Ψ stoichiometry within a targeted system used for drug development. This level of detection, achieved at single-nucleotide resolution, challenges the conventional methods of detecting Ψ. Thus, DRS is a valuable tool for the rapid and straightforward quality assessment of therapeutic RNAs,

such as mRNA vaccines and antisense oligonucleotides [49]. Additionally, DRS offers
a comprehensive single workflow for evaluating sequence identity, integrity, poly(A) tail
length, and contamination from oligonucleotides, thereby streamlining quality control
processes for therapeutic RNAs [50].

Furthermore, RNA modifications play an important role in the development of mRNA vaccines. Ψ can suppress recognition by toll-like receptors in the innate immune system. This reduces the immunogenicity of the RNA, which was a crucial breakthrough for developing effective mRNA vaccines against the SARS-Cov-2 virus with reduced side effects and improved protein translation [51], [52].  

# RNA modification detection in diagnostics  

We were also able to confirm the pathogenicity of variants in an RNA-modifying enzyme (METTL5) by predicting the m$^{6}$A stoichiometry on the 18S rRNA in a clinical patient with an intellectual developmental disorder. Besides the known disease-association of aberrant RNA-modifying functionality, there is growing evidence that dysregulation of mRNA modifications contributes to tumor development and progression, making them promising targets for future drug development [53], [54], [55], [56]. Beyond mRNA and rRNA, DRS shows growing potential for studying modifications in other RNA species, including mitochondrial RNA, tRNA, and other non-coding RNAs [16], [24], [57], [58]. These RNA types offer unique opportunities for elucidating disease mechanisms related to RNA-modification disorders [7]. Accurate prediction of molecular changes is essential for understanding disease mechanisms, and improvements in DRS accuracy with RNA004 chemistry are poised to enhance biomedical research further.  

Comprehensive analysis of the epitranscriptome will be pivotal not only for studying rare diseases but also for advancing cancer diagnostics and RNA therapies. The integration of RNA epi-signature analysis into the clinical routine screenings holds the potential to improve diagnostic precision and deepen our understanding of pathomechanisms in rare diseases. Thus, routine use of DRS in clinical settings is increasingly realistic.  

# Limitations of current RNA modification tools  

Although methodologies for detecting certain RNA modification are well established, current methodologies still face significant limitations. Despite the availability of several community-developed tools optimized for RNA002 chemistry, no existing method can comprehensively detect more than a few RNA modifications[35], [59], [60]. Such distinctions, however, are crucial for advancing our understanding of the regulatory functions of RNA modifications and their implications for health and disease. Reliable, user-friendly solutions to detecting RNA modification and their integration within standard software are especially needed for clinical applications and would facilitate better assessment and diagnosis of modopathies.  

# Model complexity and benchmarking challenges  

The development of new classification models for detecting RNA modification is ongoing. Typically, new models are benchmarked against older tools using different training  

data or chemistry, leading to significant model complexity and heterogeneity. This complexity can be overwhelming for practical use, as discussed in the one of the most recent and comprehensive reviews of m$^{6}$A base-calling models to date [61]. The review evaluated 14 m$^{6}$A-detection tools but found no universal model suitable for all applications. For example, a model trained on human cell line data performed poorly on oligonucleotide data, and vice versa.  

# Persistent error patterns in RNA004 chemistry  

Liu-Wei and colleagues investigated systematic base-calling errors in DRS and found in canonical nucleotides that, despite improvements in accuracy, the RNA004 chemistry still exhibits similar error patterns as its predecessor, such as frequent insertion and deletion errors [61]. The misbasecalls on modified nucleotides, observed in the TRID system, might arise due to the $k$-mer data used for RNA004-based training not being fully representative of all sequences. Consequently, certain sites can show base-calling errors exceeding 50%. Unfortunately, the training data and benchmarks for ONT models of RNA modification remain unavailable to the public, limiting further refinement.  

# Lack of gold standard data sets  

In our study, we have predicted > 600.000 Ψ sites using RNA004 and the Dorado basecaller, while previous publications reported only several hundred Ψ sites on mRNA level [62]. However, the overlap of the four different studies was only marginal with sensitivity and specificity being unassessed. Even minor variations in these parameters can produce disparate results when calculating overlaps [62]. Benchmarking the detection of RNA modification is further complicated by the absence of universally accepted gold-standard data sets, inconsistent sequencing depths, and diverse post-base-call filtering options.  

Another illustrative example is the comparison of called m$^{6}$A sites observed by the community-based m$^{6}$A detection tool CHEUI (CH3 (methylation) Estimation Using Ionic current) using HEK293T cell line data with the GLORI dataset, a gold standard for human m$^{6}$A sites [40], [63]. Recently, Chan and colleagues reported a site-level stoichiometry correlation with GLORI of 0.64, while the CHEUI developer itself found a correlation of 0.85 [37], [63]. The GLORI correlation in our study was around 0.86, which is closer to the second example.  

The intricacy of the RNA epitranscriptome further complicates the generation of ground-truth data sets. With over 170 known RNA modifications, it is uncertain whether each modification leaves detectable deviations in retention time or current levels. Additionally, nanopore devices detect $k$-mers, so signals are often influenced by adjacent bases within the sensing zone, potentially leading to false positives. This might explain erroneous m$^{6}$A detections at the +1 position of $\Psi$ in oligonucleotide sequences in this study. Moreover, several modifications in proximity are difficult to resolve and may require specific enzyme knockouts, which further increases data set complexity. For instance, 19 modifications in E. coli tRNAs separated by fewer than five nucleotides required methyltransferase knockdowns to isolate their signatures [64]. All in all, this makes generating ground-truth data sets challenging.  

# Toward broader clinical application of direct RNA sequencing  

For DRS to achieve widespread use in detecting RNA modifications in clinical settings, the development of gold-standard data sets for human samples, such as those established by the Genome in a Bottle (GIAB) or the Challenging Medically-Relevant Genes Benchmark-Set (CMRG), is essential [65], [66]. Another current limitation is the absence of ONT-based barcoding kits for RNA004 chemistry. This forces users to sequence an entire flow cell per sample or resort to a "nuclease flush" to remove libraries from the flow cell.  

# Conclusion  

Despite these challenges, RNA004 chemistry offers significant improvements in sequencing accuracy and throughput. Site-specific detection of modifications holds promise for integration into clinical practice, with applications extending beyond m$^{6}$A and Ψ to other modifications. Potential uses include site-specific assays and quality control of RNA therapeutics, as we could demonstrate in our paper given the TRID system and the METTL5 case. The growing number of RNA004 users could provide the impetus to close these gaps and ultimately realize the potential of DRS to enrich clinical care and diagnostics.  

# Ethical statement  

This is a basic research project to validate direct RNA sequencing to evaluate its suitability for detecting molecular targets for innovative forms of therapy. In addition to a standardized sample (blood from a healthy volunteer), we also examined blood from an infant with an autosomal recessive intellectual developmental disorder type 72 showing a putative splice site on METTL5 and the observed reduction of m$^{6}$A at the A1832 position of 18S rRNA in the patient. Since the infant was unable to understand the aims, scope, risks, and benefits of the study, and because we are reporting on a rare disorder, the patient was considered highly vulnerable. Thus, informing the parents as legal proxies about all aspects of the rather complex procedure was paramount to safeguard the interests of the patient. The diagnosis and the associated functional consequence were determined in one assay using nanopore sequencing. Regarding its suitability for the detection of molecular markers, artificial modifications of RNA were also used to test the stability, sensitivity, and selectivity of the method for identifying pathologically relevant molecular targets.  

The project was evaluated by the internal ethics advisory board of the University Medical Centre. From an ethical point of view, this is basic research without direct reference to patient care. Informed consent was obtained from the legal proxies that surplus material (blood) was intended to be used for the validation of a new method for direct RNA sequencing. Data was anonymized and the risk of reference back to individuals due to the processing of genetic information in the case of rare disease was pointed out, as well as the fact that no whole genome data was generated or analyzed. However, both the proband and legal proxies consider the possible future risk to be acceptable when weighed against the gain in knowledge.  

The research presented here is explicitly not a clinical study. The study was therefore evaluated by the internal ethics advisory board. Ethical principles, in particular the principle of autonomy, are upheld, which is especially true in light of the revision of the Declaration of Helsinki, which aims to enable research in this area while maintaining the protection of vulnerable groups such as children in order to facilitate access to innovative medical procedures. This also applies in the case of the present study to validate the clinical applicability of new diagnostic procedures or the identification of molecular targets, even if there is currently no direct

patient benefit but at most a group benefit. This study, using a single sample of one vulnerable patient providing relevant information diligently and obtaining fully informed consent of the legal proxies to validate a novel diagnostic strategy does not raise ethical concerns. However, 822 should the concept be translated into a (translational) clinical study, ethical approval would 823 have to be obtained by the regulatory authorities.

# Data Availability  

The data for this study have been deposited in the European Nucleotide Archive (ENA) at EMBL-EBI under the accession number PRJEB74238. The human phenotype data will be deposited to EGA once the manuscript has been conditionally accepted.  

Code Availability All code written in support of this publication is publicly available at https://github.com/CSG-Group-Mainz/RNA004-Manuscript.  

Acknowledgements: This work was partly funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation; project no. 439669440 TRR319 RMaP TP A01/A05/C01/C03 to F. L., J. K., M.H. and S.M). S.W. and S.G. acknowledge funding from the Emergent AI Center funded by the Carl-Zeiss-Stiftung. S.S. and S.G. acknowledge funding from the Forschungsinitiative Rheinland-Pfalz and the ReALity initiative of the Johannes Gutenberg University Mainz. S.Sy. acknowledges the M3odel initiative from the Forschungsinitiative Rheinland-Pfalz. This work was also partly supported by funding from ERC ADG Multi-OrganelleDesign (E.A.L.). S.G. and C.H. acknowledge funding from the Boehringer Ingelheim Stiftung.  

F. L. and J. K. thank the Next Generation Sequencing Core Facility of the German Cancer Research Center, particularly Franziska Petermann and Panagiotis Provataris for their support.  

Author Contributions: C.H. designed the project, wrote the manuscript, and performed data analysis. A.W. performed data analysis, wrote the manuscript, and composed the figures. S.D., V.H., F.K., and L.H. supported with patient recruitment and clinical interpretation of the variants. T.B. performed the sequencing of the cell line data. J.F. sequenced the peripheral blood samples supported by K.J. S.M. sequenced the oligos under the supervision of M.H. J.M., S.S., V.D., K.B., S.W., and F.H. contributed to data analysis, to writing the manuscript and designed parts of the figures. L.S. designed the TRID system and performed the analysis under the supervision of E.A.L. J.K. prepared the GLORI sequencing data under the supervision of F. L. S.G., and M.L. supervised the study, edited the manuscript and contributed to writing and conceptualizing the manuscript. All authors approved and proofread the manuscript.  

# References  

[1] A. Cappannini et al., "MODOMICS: a database of RNA modifications and related information. 2023 update," Nucleic Acids Res., vol. 52, no. D1, pp. D239–D244, Nov. 2024.

[2] R. Karthiya and P. Khandelia, "m6A RNA Methylation: Ramifications for Gene Expression and Human Health," Mol. Biotechnol., vol. 62, no. 10, pp. 467–484, Oct. 2020, doi: 10.1007/s12033-020-00269-5.  

[3] N. M. Martinez et al., "Pseudouridine synthases modify human pre-mRNA co-transcriptionally and affect pre-mRNA processing," Mol. Cell, vol. 82, no. 3, pp. 645-659.e9, Nov. 2022, doi: 10.1016/j.molcel.2021.12.023.

[4] J. Karijolich and Y.-T. Yu, "Converting nonsense codons into sense codons by targeted pseudouridylation," Nature, vol. 474, no. 7351, pp. 395-398, Nov. 2011.

[5] I. Toledano, F. Supek, and B. Lehner, "Genome-scale quantification and prediction of pathogenic stop codon readthrough by small molecules," Nat. Genet., vol. 56, no. 9, pp. 1914-1924, Nov. 2024.

[6] L. Schartel et al., "Selective {RNA} pseudouridinylation in situ by circular {gRNAs} in designer organelles," Nat. Commun., vol. 15, no. 1, p. 9177, Nov. 2024.

[7] S. Delaunay, M. Helm, and M. Frye, "{RNA} modifications in physiology and disease: towards clinical applications," Nat. Rev. Genet., vol. 25, no. 2, pp. 104-122, Nov. 2024.

[8] N. Jonkhout, J. Tran, M. A. Smith, N. Schonrock, J. S. Mattick, and E. M. Novoa, "The RNA modification landscape in human disease," Nov. 2017, Cold Spring Harbor Laboratory Press. doi: 10.1261/rna.063503.117.

[9] T. Suzuki, "The expanding world of tRNA modifications and their disease relevance," Nov. 2021, Nature Research. doi: 10.1038/s41580-021-00342-0.

[10] J. E. Mangum et al., "Pseudouridine synthase 1 deficient mice, a model for Mitochondrial Myopathy with Sideroblastic Anemia, exhibit muscle morphology and physiology alterations," Sci. Rep., vol. 6, no. 1, Nov. 2016.

[11] R. Shaheen et al., "PUS7 mutations impair pseudouridylation in humans and cause intellectual disability and microcephaly," Hum. Genet., vol. 138, no. 3, pp. 231-239, Nov. 2019, doi: 10.1007/s00439-019-01980-3.

[12] M. Nøstvik et al., "Clinical and molecular delineation of PUS3-associated neurodevelopmental disorders," Clin. Genet., vol. 100, no. 5, pp. 628-633, Nov. 2021, doi: 10.1111/cge.14051.

[13] S. Schwartz et al., "Transcriptome-wide Mapping Reveals Widespread Dynamic-Regulated Pseudouridylation of ncRNA and mRNA," Cell, vol. 159, no. 1, pp. 148-162, Sep. 2014, doi: 10.1016/j.cell.2014.08.028.

[14] M. Taoka et al., "Landscape of the complete RNA chemical modifications in the human 80S ribosome," Nucleic Acids Res., vol. 46, no. 18, pp. 9289-9298, Nov. 2018, doi: 10.1093/nar/gky811.

[15] R. Gao, M. Ye, B. Liu, M. Wei, D. Ma, and K. Dong, "m6A Modification: A Double-Edged Sword in Tumor Development," Front. Oncol., vol. 11, Jul. 2021, doi: 10.3389/fonc.2021.679367.

[16] I. Milenkovic et al., "Epitranscriptomic rRNA fingerprinting reveals tissue-of-origin and tumor-specific signatures," Nov. 2024, Cold Spring Harbor Laboratory. doi: 10.1101/2024.10.03.616461.

[17] E. M. Turkkalj and C. Vissers, "The emerging importance of METTL5-mediated ribosomal RNA methylation," Nov. 2022, Springer Nature. doi: 10.1038/s12276-022-00869-y.  

[18] E. M. Richard et al., "Bi-allelic variants in {METTL5} cause autosomal-recessive intellectual disability and microcephaly," Am. J. Hum. Genet., vol. 105, no. 4, pp. 869–878, Nov. 2019.

[19] C. Soneson, M. I. Love, and M. D. Robinson, "Differential analyses for RNA-seq: Transcript-level estimates improve gene-level inferences," F1000Research, vol. 4, p. 1521, Nov. 2016, doi: 10.12688/F1000RESEARCH.7563.2.

[20] J. D. Alfonzo, J. A. Brown, P. H. Byers, V. G. Cheung, R. J. Maria, and R. L. Ross, "A call for direct sequencing of full-length RNAs to identify all modifications," Nat. Genet., vol. 53, no. 8, pp. 1113–1116, Nov. 2021.

[21] D. R. Garalde et al., "Highly parallel direct {RNA} sequencing on an array of nanopores," Nat. Methods, vol. 15, no. 3, pp. 201–206, Nov. 2018.

[22] M. Helm and Y. Motorin, "Detecting {RNA} modifications in the epitranscriptome: predict and validate," Nat. Rev. Genet., vol. 18, no. 5, pp. 275–291, Nov. 2017.

[23] Y. Motorin and M. Helm, "{RNA}nucleotide} methylation: 2021 update," WIREs RNA, vol. 13, 2022.

[24] M. C. Lucas et al., "Quantitative analysis of {tRNA} abundance and modifications by nanopore {RNA} sequencing," Nat. Biotechnol., vol. 42, no. 1, pp. 72–86, Nov. 2023.

[25] C. Soneson, Y. Yao, A. Bratus-Neuenschwander, A. Patrignani, M. D. Robinson, and S. Hussain, "A comprehensive examination of Nanopore native RNA sequencing for characterization of complex transcriptomes," Nat. Commun., vol. 10, no. 1, Nov. 2019, doi: 10.1038/s41467-019-11272-z.

[26] M. Jain, R. Abu-Shumays, H. E. Olsen, and M. Akeson, "Advances in nanopore direct {RNA} sequencing," Nat. Methods, vol. 19, no. 10, pp. 1160–1164, Nov. 2022.

[27] M. Sereika et al., "Oxford Nanopore R10.4 long-read sequencing enables the generation of near-finished bacterial genomes from pure cultures and metagenomes without short-read or reference polishing," Nat. Methods, vol. 19, no. 7, pp. 823–826, Nov. 2022, doi: 10.1038/s41592-022-01539-7.

[28] R. R. Wick, L. M. Judd, and K. E. Holt, "Performance of neural network basecalling tools for Oxford Nanopore sequencing," Genome Biol., vol. 20, no. 1, p. 129, Nov. 2019, doi: 10.1186/s13059-019-1727-y.

[29] SEQC/MAQC-III Consortium, "A comprehensive assessment of RNA-seq accuracy, reproducibility and information content by the Sequencing Quality Control Consortium," Nat. Biotechnol., vol. 32, no. 9, pp. 903–914, Sep. 2014, doi: 10.1038/nbt.2957.

[30] A. Brouze, P. S. Krawczyk, A. Dziembowski, and S. Mrozek, "Measuring the tail: Methods for poly(A) tail profiling," Nov. 2023, John Wiley and Sons Inc. doi: 10.1002/wrna.1737.

[31] G. Pagani and P. Gandellini, "Cleavage and polyadenylation machinery as a novel targetable vulnerability for human cancer," Cancer Gene Ther., vol. 31, no. 7, pp. 957–960, Nov. 2024.  

[32] M. Krause, A. M. Niazi, K. Labun, Y. N. T. Cleuren, F. S. Müller, and E. Valen, "tailfindr: alignment-free poly(A) length measurement for Oxford Nanopore {RNA} and {DNA} sequencing," RNA, vol. 25, no. 10, pp. 1229–1241, Nov. 2019.

[33] O. Begik et al., "Quantitative profiling of pseudouridylation dynamics in native RNAs with nanopore sequencing," Nat. Biotechnol., vol. 39, no. 10, pp. 1278–1291, Nov. 2021, doi: 10.1038/s41587-021-00915-6.

[34] M. Furlan, A. Delgado-Tejedor, L. Mulroney, M. Pelizzola, E. M. Novoa, and T. Leonardi, "Computational methods for RNA modification detection from nanopore direct RNA sequencing data," Nov. 2021, Taylor and Francis Ltd. doi: 10.1080/15476286.2021.1978215.

[35] A. Leger et al., "RNA modifications detection by comparative Nanopore direct RNA sequencing," Nat. Commun., vol. 12, no. 1, p. 7198, Nov. 2021, doi: 10.1038/s41467-021-27393-3.

[36] T. A. Nguyen et al., "Direct identification of A-to-I editing sites with nanopore native RNA sequencing," Nat. Methods, vol. 19, no. 7, pp. 833–844, Nov. 2022, doi: 10.1038/s41592-022-01513-3.

[37] A. Chan, I. S. N. Vries, C. P. M. Scheitl, C. Hübartner, and C. Dieterich, "Detecting m6A at single-molecular resolution via direct {RNA} sequencing and realistic training data," Nat. Commun., vol. 15, no. 1, p. 3323, Nov. 2024.

[38] A. Chan, I. S. Naarmann-de Vries, C. P. M. Scheitl, C. Hübartner, and C. Dieterich, "Detecting m6A at single-molecular resolution via direct RNA sequencing and realistic training data," Nat. Commun., vol. 15, no. 1, p. 3323, Apr. 2024, doi: 10.1038/s41467-024-47661-2.

[39] S. Cruciani, A. Delgado-Tejedor, L. P. Pryszcz, R. Medina, L. Llovera, and E. M. Novoa, "De novo basecalling of m6A modifications at single molecule and single nucleotide resolution," Nov. 2023, [Online]. Available: https://doi.org/10.1101/2023.11.13.566801

[40] C. Liu et al., "Absolute quantification of single-base m6A methylation in the mammalian transcriptome using {GLORI},” Nat. Biotechnol., vol. 41, no. 3, pp. 355–366, Nov. 2023.

[41] W. Shen et al., "GLORI for absolute quantification of transcriptome-wide m6A at single-base resolution," Nat. Protoc., vol. 19, no. 4, pp. 1252–1287, Nov. 2024, doi: 10.1038/s41596-023-00937-1.

[42] S. Tavakoli et al., "Semi-quantitative detection of pseudouridine modifications and type I/II hypermodifications in human mRNAs using direct long-read sequencing," Nat. Commun., vol. 14, no. 1, pp. 1–12, Nov. 2023, doi: 10.1038/s41467-023-35858-w.

[43] Z. Guo et al., "Nanopore Current Events Magnifier (nanoCEM): a novel tool for visualizing current events at modification sites of nanopore sequencing," NAR Genomics Bioinforma., vol. 6, no. 2, Nov. 2024, doi: 10.1093/nargab/lqae052.

[44] Q. Dai et al., "Quantitative sequencing using {BID-seq} uncovers abundant pseudouridines in mammalian {mRNA} at base resolution," Nat. Biotechnol., vol. 41, no. 3, pp. 344–354, Nov. 2023.  

[45] R. E. Workman et al., "Nanopore native RNA sequencing of a human poly(A) transcriptome," Nat. Methods, vol. 16, no. 12, pp. 1297–1305, Dec. 2019, doi: 10.1038/s41592-019-0617-2.

[46] Y. Chen et al., "Context-aware transcript quantification from long-read {RNA-seq} data with Bambu," Nat. Methods, vol. 20, no. 8, pp. 1187–1195, Nov. 2023.

[47] C. Ugolini et al., "Nanopore ReCappable sequencing maps SARS-CoV-2 5′ capping sites and provides new insights into the structure of sgRNAs," Nucleic Acids Res., vol. 50, no. 6, pp. 3475–3489, Nov. 2022, doi: 10.1093/nar/gkac144.

[48] F. J. Pardo-Palacios et al., "Systematic assessment of long-read RNA-seq methods for transcript identification and quantification," Nat. Methods, vol. 21, no. 7, pp. 1349–1363, Nov. 2024, doi: 10.1038/s41592-024-02298-3.

[49] F. Conroy et al., "Chemical engineering of therapeutic {siRNAs} for allele-specific gene silencing in Huntington’s disease models," Nat. Commun., vol. 13, no. 1, p. 5802, Nov. 2022.

[50] H. M. Gunter et al., "mRNA vaccine quality analysis using RNA sequencing," Nat. Commun., vol. 14, no. 1, p. 5663, Nov. 2023.

[51] K. Karikó, M. Buckstein, H. Ni, and D. Weissman, "Suppression of {RNA} recognition by Toll-like receptors: the impact of nucleoside modification and the evolutionary origin of {RNA},” Immunity, vol. 23, no. 2, pp. 165–175, Nov. 2005.

[52] U. Sahin et al., "COVID-19 vaccine BNT162b1 elicits human antibody and TH1 T cell responses," Nature, vol. 586, no. 7830, pp. 594–599, Oct. 2020, doi: 10.1038/s41586-020-2814-7.

[53] M. Cieśla et al., "m6A-driven {SF3B1} translation control steers splicing to direct genome integrity and leukemogenesis," Mol. Cell, vol. 83, no. 7, pp. 1165–1179.e11, Nov. 2023.

[54] Q. Cui et al., "Targeting {PUS7} suppresses {tRNA} pseudouridylation and glioblastoma tumorigenesis," Nat. Cancer, vol. 2, no. 9, pp. 932–949, Nov. 2021.

[55] D. Dai, H. Wang, L. Zhu, H. Jin, and X. Wang, "N6-methyladenosine links {RNA} metabolism to cancer progression," Cell Death Dis., vol. 9, no. 2, Nov. 2018.

[56] J. Wang et al., "Leukemogenic Chromatin Alterations Promote AML Leukemia Stem Cells via a KDM4C-ALKBH5-AXL Signaling Axis," Cell Stem Cell, vol. 27, no. 1, pp. 81–97.e8, Jul. 2020, doi: 10.1016/j.stem.2020.04.001.

[57] M. Jörg et al., "N1-methylation of adenosine (m1A) in {ND5} {mRNA} leads to complex {I} dysfunction in Alzheimer’s disease," Mol. Psychiatry, vol. 29, no. 5, pp. 1427–1439, Nov. 2024.

[58] K. Nemeth, R. Bayraktar, M. Ferracin, and G. A. Calin, "Non-coding RNAs in disease: from mechanisms to therapeutics," Nov. 2024, Nature Research. doi: 10.1038/s41576-023-00662-1.

[59] H. Teng, M. Stoiber, Z. Bar-Joseph, and C. Kingsford, "Detecting m6A RNA modification from nanopore sequencing using a semisupervised learning framework," Genome Res., vol. 34, no. 11, pp. 1987–1999, Nov. 2024, doi: 10.1101/gr.278960.124.  

[60] S. Cruciani, A. Delgado-Tejedor, L. P. Pryszcz, R. Medina, L. Llovera, and E. M. Novoa, “\textit{De novo}basecalling of {m$^{6}$A} modifications at single molecule and single nucleotide resolution,” Nov. 2023.

[61] S. Maestri et al., “Benchmarking of computational methods for m6A profiling with Nanopore direct {RNA} sequencing,” Brief. Bioinform., vol. 25, no. 2, Nov. 2024.

[62] T.-Y. Lin et al., “The molecular basis of {tRNA} selectivity by human pseudouridine synthase 3,” Mol. Cell, vol. 84, no. 13, pp. 2472–2489.e8, Nov. 2024.

[63] P. Acera Mateos et al., “Prediction of m6A and m5C at single-molecule resolution reveals a transcriptome-wide co-occurrence of RNA modifications,” Nat. Commun., vol. 15, no. 1, pp. 1–17, Dec. 2024, doi: 10.1038/s41467-024-47953-7.

[64] A. M. Fleming, N. J. Mathewson, S. A. H. Manage, and C. J. Burrows, “Nanopore Dwell Time Analysis Permits Sequencing and Conformational Assignment of Pseudouridine in SARS-CoV-2,” ACS Cent. Sci., vol. 7, no. 10, pp. 1707–1717, Nov. 2021, doi: 10.1021/acscentsci.1c00788.

[65] J. M. Zook et al., “A robust benchmark for detection of germline large deletions and insertions,” Nat. Biotechnol., vol. 38, no. 11, pp. 1347–1355, Nov. 2020, doi: 10.1038/s41587-020-0538-8.

[66] J. Wagner et al., “Curated variation benchmarks for challenging medically relevant autosomal genes,” Nat. Biotechnol., vol. 40, no. 5, pp. 672–680, Nov. 2022, doi: 10.1038/s41587-021-01158-1.

[67] S. S. George, M. Pimkin, and V. R. Paralkar, “Construction and validation of customized genomes for human and mouse ribosomal DNA mapping,” J. Biol. Chem., vol. 299, no. 6, Nov. 2023, doi: 10.1016/j.jbc.2023.104766.  
