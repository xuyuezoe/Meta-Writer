---
paper_id: cfd29cbb530e4e4ca2bc8441a50ae9ce
doi: 10.1101/2024.12.18.629302
source: biorxiv
version_date: '2025-01-04'
license: CC-BY 4.0
title: Mettl15-Mettl17 modulates the transition from early to late pre-mitoribosome
authors:
- name: Yury Zgadzay
  affiliations:
  - 1
  - 2
  corresponding: false
  email: null
- name: Claudio Mirabello
  affiliations:
  - 3
  corresponding: false
  email: null
- name: George Wanes
  affiliations:
  - 5
  - 6
  corresponding: false
  email: null
- name: Tomáš Pánek
  affiliations:
  - 7
  corresponding: false
  email: null
- name: Prashant Chauhan
  affiliations:
  - 8
  - 9
  corresponding: false
  email: null
- name: Björn Nystedt
  affiliations:
  - 4
  corresponding: false
  email: null
- name: Alena Zíková
  affiliations:
  - 8
  - 9
  corresponding: false
  email: null
- name: Paul C. Whitford
  affiliations:
  - 5
  - 6
  corresponding: false
  email: null
- name: Ondřej Gahura
  affiliations:
  - 9
  corresponding: true
  email: gahura@paru.cas.cz
- name: Alexey Amunts
  affiliations:
  - 2
  corresponding: true
  email: alexey.amunts@gmail.com
affiliations:
  1: Department of Integrated Structural Biology, Institute of Genetics and Molecular and Cellular Biology, University of
    Strasbourg, Illkirch, France.
  2: Science for Life Laboratory, Department of Biochemistry and Biophysics, Stockholm University, 17165 Solna, Sweden
  3: Dept of Physics, Chemistry and Biology, National Bioinformatics Infrastructure Sweden, Science for Life Laboratory, Linköping
    University, 581 83 Linköping, Sweden
  4: Dept of Cell and Molecular Biology, National Bioinformatics Infrastructure Sweden, Science for Life Laboratory, Uppsala
    University, Husargatan 3, SE-752 37 Uppsala, Sweden
  5: Department of Physics, Northeastern University, Boston, MA, 02115, USA
  6: Center for Theoretical Biological Physics, Northeastern University, Boston, MA, 02115, USA
  7: Department of Zoology, Faculty of Science, Charles University, Prague, Czech Republic
  8: Faculty of Science, University of South Bohemia, 37005 České Budějovice, Czech Republic
  9: Institute of Parasitology, Biology Centre, Czech Academy of Sciences, 37005 České Budějovice, Czech Republic
abstract: The assembly of the mitoribosomal small subunit involves folding and modification of rRNA, and its association with
  mitoribosomal proteins. This process is assisted by a dynamic network of assembly factors. Conserved methyltransferases
  Mettl15 and Mettl17 act on the solvent-exposed surface of rRNA. Binding of Mettl17 is associated with the early assembly
  stage, whereas Mettl15 is involved in the late stage, but the mechanism of transition between the two was unclear. Here,
  we integrate structural data from Trypanosoma brucei with mammalian homologs and molecular dynamics simulations. We reveal
  how the interplay of Mettl15 and Mettl17 in intermediate steps links the distinct stages of small subunit assembly. The
  analysis suggests a model wherein Mettl17 acts as a platform for Mettl15 recruitment. Subsequent release of Mettl17 allows
  a conformational change of Mettl15 for substrate recognition. Upon methylation, Mettl15 adopts a loosely bound state which
  ultimately leads to its replacement by initiation factors, concluding the assembly. Together, our results indicate that
  assembly factors Mettl15 and Mettl17 cooperate to regulate the biogenesis process.
keywords: null
paper_type: research-article
subject_areas:
- Structural Biology
- Molecular Biology
- Biochemistry
- Mitochondrial Biology
datasets: null
stats:
  word_count: 8377
  has_math: true
  section_count: 21
---
# Mettl15-Mettl17 modulates the transition from early to late pre-mitoribosome

# Mettl15-Mettl17 modulates the transition from early to late pre-mitoribosome  

Yury Zgadzay$^{1,2}$, Claudio Mirabello$^{3,\#}$, George Wanes$^{5,6,\#}$, Tomáš Pánek$^{7}$, Prashant Chauhan$^{8,9}$,

Björn Nystedt$^{4}$, Alena Zíková$^{8,9}$, Paul C. Whitford$^{5,6}$, Ondřej Gahura$^{9,*}$, Alexey Amunts$^{2,*}$

7 $^{1}$ Department of Integrated Structural Biology, Institute of Genetics and Molecular and Cellular
Biology, University of Strasbourg, Illkirch, France.

² Science for Life Laboratory, Department of Biochemistry and Biophysics, Stockholm University, 17165 Solna, Sweden  

³ Dept of Physics, Chemistry and Biology, National Bioinformatics Infrastructure Sweden, Science for Life Laboratory, Linköping University, 581 83 Linköping, Sweden  

$^{4}$ Dept of Cell and Molecular Biology, National Bioinformatics Infrastructure Sweden, Science for Life Laboratory, Uppsala University, Husargatan 3, SE-752 37 Uppsala, Sweden  

$^{5}$ Department of Physics, Northeastern University, Boston, MA, 02115, USA  

$^{6}$ Center for Theoretical Biological Physics, Northeastern University, Boston, MA, 02115, USA  

${}^{7}$ Department of Zoology, Faculty of Science, Charles University, Prague, Czech Republic  

$^{8}$Faculty of Science, University of South Bohemia, 37005 České Budějovice, Czech Republic  

$^{9}$ Institute of Parasitology, Biology Centre, Czech Academy of Sciences, 37005 České Budějovice, Czech Republic  

# Equal contribution  

* Correspondence to: gahura@paru.cas.cz; alexey.amunts@gmail.com  

# ABSTRACT  

The assembly of the mitoribosomal small subunit involves folding and modification of rRNA, and its association with mitoribosomal proteins. This process is assisted by a dynamic network of assembly factors. Conserved methyltransferases Mettl15 and Mettl17 act on the solvent-exposed surface of rRNA. Binding of Mettl17 is associated with the early assembly stage, whereas Mettl15 is involved in the late stage, but the mechanism of transition between the two was unclear. Here, we integrate structural data from Trypanosoma brucei with mammalian homologs and molecular dynamics simulations. We reveal how the interplay of Mettl15 and Mettl17 in intermediate steps links the distinct stages of small subunit assembly. The analysis suggests a model wherein Mettl17 acts as a platform for Mettl15 recruitment. Subsequent release of Mettl17 allows a conformational change of Mettl15 for substrate recognition. Upon methylation, Mettl15 adopts a loosely bound state which ultimately leads to its replacement by initiation factors, concluding the assembly. Together, our results indicate that assembly factors Mettl15 and Mettl17 cooperate to regulate the biogenesis process.  

In mitochondria, messenger RNA (mRNA) translation and protein synthesis are performed by the mitoribosome in association with the regulatory complex LRPPRC-SLIRP$^{1}$ and the OXA1L insertase at the inner mitochondrial membrane$^{2,3}$. The mammalian mitoribosome consists of three mitochondria-encoded RNA molecules with 19 modified nucleotides and at least 82 nuclear-encoded proteins$^{4-6}$. The formation of this complex machinery involves progressive assembly through the recruitment of assembly factors that act primarily on the ribosomal RNA (rRNA), triggering its gradual folding and modification, while mitoribosomal protein modules are formed$^{7-9}$. This allows for productive maturation through defined states that ultimately leads to the catalytic mitoribosome$^{10}$. Perturbations in the assembly pathway can underlie association of mitoribosomal dysfunction with various diseases$^{11-14}$.  

Structural studies on Trypanosoma brucei mitoribosomal complexes showed that it is a good model for understanding fundamental principles of mitoribosomal assembly because its native pre-mitoribosomal complexes are biochemically more stable and contain most of the assembly factors observed in mammals and other eukaryotes$^{15-19}$. For example, T. brucei mitoribosomal large subunit biogenesis involves at least seven assembly factors shared with humans, including GTPases GTPBP7, MTG1, pseudouridinase RPUSD4, and methyltransferase MRM. The structures of intermediates with these factors allowed for a better understanding of their roles in the mitoribosomal assembly pathway$^{17,18}$.  

The mammalian mitoribosomal small subunit (mtSSU) is highly dynamic and contains 12S rRNA with nicotinamide adenine dinucleotide associated with an rRNA insertion, 30 mitoribosomal proteins, and two iron-sulfur (Fe-S) clusters$^{20-23}$. The structure is arranged into two main regions defined as the body and head. The latter binds LRPPRC-SLIRP, which regulates mRNA delivery to a dedicated channel during translation initiation and undergoes conformational changes to accompany the movement of mRNA during translation cycle$^{4,5,24}$. In skeletal muscle, exercise training-induced signalling leads to enhanced mitoribosomal activity that can bypass LRPPRC-SLIRP$^{25}$. The assembly path involves at least 11 factors that facilitate binding of mitoribosomal proteins and construct the solvent-exposed surface of the rRNA, including the mRNA channel and the decoding centre at the interface between the body and head$^{8}$.  

Structural studies have revealed that stable assembly intermediates of the small subunit can be divided into 'early'$^{26}$ and 'late'$^{22}$ stages, each relying on distinct methyltransferases, Mettl17 and Mettl15, respectively. Mettl17 is also an Fe-S binding protein that serves as a checkpoint for mitochondrial translation$^{27}$. However, the transition from the early to late stage, including  

the interaction between Mettl15 and Mettl17, has never been observed. It is also not clear what drives the release of Mettl17, which promotes maturation, primarily due to limitations in the experimental design.  

Studying mitoribosomal assembly intermediates experimentally is challenging because interactions between assembly factors are often dynamic, and the transient states can undergo dissociation when isolated for structural analysis. In addition, adding tags to assembly factors for protein purification can interfere with native interactions, and knockout strains might exhibit non-productive off-path configurations of pre-mitoribosome, thus compromising the interpretation. However, the recent development of new computational tools for the analysis of protein-protein interactions$^{28,29}$ enabled studies on large nucleoprotein complexes involved in gene expressions and associated with transient modifying enzymes$^{23,24}$. Thus, in silico approaches can reveal direct interacting partners and propose models of sequential assembly of macromolecular complexes.  

Here, we used the cryo-EM map of T. brucei mtSSU assembly intermediate$^{15}$ and structural models of human early$^{26}$ and late$^{22}$ intermediates. Leveraging recent computational advancements, we performed AlphaFold2$^{30}$ analysis and molecular dynamics simulations, to generate in silico models for previously undescribed states. This approach enabled us to propose a sequential mechanism that explains the structural basis for the Mettl15-Mettl17 function on the pre-mitoribosome.  

# RESULTS  

# Mapping unassigned regions in the pre-mitoribosome uncovers Mettl15-Mettl17 heterodimer  

A mtSSU intermediate from T. brucei has been previously studied by cryo-EM, but several regions in the map remained unassigned$^{15}$. Using the data from recently published structures of mammalian pre-mitoribosomal intermediates$^{22,26}$, we analysed the T. brucei maps and identified a number of previously undescribed structural elements (Table 1).  

First, we detected homologs of RbfA and Mettl15 (previously referred to as mt-SAF18 and mt-SAF14, respectively), both of which are associated with Mettl17 (mt-SAF1) (Fig. 1A). RbfA is a KH-domain containing assembly factor (Fig. 1B) that scaffolds decoding center rRNA elements, contacts the 3'end of rRNA, and occupies the mRNA channel during ribosomal assembly in bacteria and mitochondria$^{22,31-34}$. Mettl15 is a class I SAM-dependent N4-  

methylcytidine (m$^{4}$C) methyltransferase of bacterial origin that modifies the mtSSU rRNA at position C1486 (human numbering)$^{35-38}$. Mettl17, in contrast, is a putative methyltransferase with no specific target$^{39,40}$, and the disruption of its interaction with the pre-mitoribosome impairs other methyltransferases as well$^{35,36,38}$. Structurally, RbfA is anchored to the complex by its N-terminal extension, with the C-terminal domain binding Mettl17 and the C-terminal extension binding Mettl15. Together, these elements stabilize the subcomplex in a way that Mettl15 and Mettl17 form a heterodimer that is bound in the cleft between the head and body (Fig. 1C). The Mettl15-Mettl17 heterodimer has the Complexation Significance Score of 0.695, this score is defined as the maximal fraction of the total free energy of binding$^{61}$, which indicates a specific interface, and the interaction surface area is 4380 Å$^{2}$. In total there are 43 hydrogen bonds and 10 salt bridges that stabilize the Mettl15-Mettl17 heterodimer (Supplementary Table 1), which contribute to two main interfaces. The first interface comprises the N-terminal part of Mettl17 (residues 45-95) and C-terminal part of Mettl15 (residues 402-470). The second interface involves catalytic domains of both Mettl17 and Mettl15 (residues 484-512 and 176-198, respectively). These data suggest that in T. brucei Mettl15 and Mettl17 form a stable complex, which has not been observed in other species.  

Both methyltransferases in the structure contain a functional prosthetic group S-adenosyl methionine (SAM) (Fig. 1C). In Mettl15, SAM is located 42 Å away from the methyltransferase target residue cytidine 582 (C582, equivalent of human C1486), implying a non-catalytic conformation. The Mettl15 conformation is different from that observed in the mammalian m$^{4}$C1486-containing post-catalytic precursor$^{22}$, thus implying a pre-catalytic state. In Mettl17, there is a density corresponding to an iron-sulphur Fe$_{4}$S$_{4}$ cluster, consistent with the mammalian$^{26}$ and yeast$^{27}$ homologs. Thus, the previously proposed role of Mettl17 as an oxidative stress sensor and an Fe-S checkpoint for mitochondrial translation$^{27,41}$ may be conserved in a broad range of eukaryotes.  

In addition, we assigned an uninterpreted region of density to trypanosomal assembly factor mt-SAF38 (Extended Data Fig. 1A). Its overall fold is similar to a thioesterase, expanding a list of enzyme homologs identified in mitoribosomal subunits or their precursors$^{42}$. Finally, we identified 13 hammerhead-shaped densities ranging from 17 to 22 Å in length, coordinated by tryptophan residues within a helix-loop-helix motif of pentatricopeptide repeat (PPR) proteins, which likely represent cofactors such as acetyl coenzyme A (acetyl-CoA) (Extended Data Fig. 1B).  

# Evolutionary conservation of Mettl17 suggests its role in recruiting Mettl15  

To determine whether the Mettl15-Mettl17 heterodimer is a group-specific feature or may be widespread, we searched for these two methyltransferases in genomes of diverse eukaryotic organisms, followed by phylogenetic analysis. The search identified Mettl17 in out of organisms covering all major eukaryotic lineages. The cysteine residues coordinating the Fe₄S₄ cluster in mammals and trypanosomes are conserved in most identified Mettl17 homologs.  

While Mettl15 is present in fewer organisms, it was identified in nearly all species where Mettl17 was present (Fig. 2, Supplementary Data 1&2, Supplementary Figs. 1&2). This suggests that Mettl17 may be a prerequisite for the incorporation of Mettl15 into the pre-mitoribosome. Mettl17 is essential for mitochondrial translation in human cells$^{39}$, for mitoribosomal assembly, translation and viability in T. brucei$^{15,43}$, and for respiration in budding yeast$^{44}$, but there is currently no evidence for the methyltransferase activity of this protein in any organism. Instead, human Mettl17 is required for methylation of rRNA by Mettl15$^{39}$. Thus, consistently with other enzymes that adopted a structural role in the mitoribosome$^{45}$, Mettl17 is an essential and conserved protein with no specific methylation target, whose primary function may be to facilitate Mettl15 integration into the pre-mitoribosome.  

Mettl15 associates with Mettl17 on the pre-mitoribosome during early assembly stage

To clarify at which stage Mettl15 associates with Mettl17 on the pre-mitoribosome, we used structural models of human early$^{26}$ and late$^{22}$ intermediate as references. The early intermediate contains Mettl17 and another methyltransferase, TFB1M (PDB ID 8CSP), whereas the late stage contains Mettl15 in a different conformation (PDB ID 7PNX). We generated AlphaFold2 (AF2)$^{28}$ models of human Mettl17-TFB1M and Mettl15-TFB1M (Fig. 3A,B). The two models obtained similar protein interface (ipTM) scores of 0.66 and 0.59, respectively, which would indicate reasonable confidence, according to the most recent benchmarking of AF prediction of multi-chain protein complexes$^{46}$. The AF2 model of Mettl17-TFB1M corresponds to the experimental dimer of the two proteins in the cryo-EM structure of the early state$^{26}$, supporting the computational approach. We then used TFB1M as an anchoring point for superposition of Mettl15-TFB1M from the predicted model onto the early intermediate (Fig. 3C). The superposition shows that Mettl15 is compatible with Mettl17, except minor clashes observed between a loop in Mettl17 (residues 220-247) and Mettl15 (residues 205-249). However, the Mettl17 loop has relatively high B-factor compared to rigid parts of the protein in current structures (Extended Data Figure 2), indicating it is rather flexible and could attain alternative conformations when in complex with Mettl15. This suggests that Mettl15 could be structurally co-localized with Mettl17, TFB1M and RbfA on the pre-mitoribosome. This is further supported by biochemical evidence, as TFB1M readily co-immunoprecipitate with Mettl15$^{36}$. Thus, Mettl15 potentially associates with the pre-mitoribosome during the early assembly stage, possibly co-constituting a state with all three methyltransferases bound (Fig. 3C).  

# Pre-mitoribosome with Mettl15 and Mettl17 represents a pre-catalytic state  

To establish the context of the pre-mitoribosome for the association of Mettl15 with Mettl17, we constructed and refined a model of the human pre-mitoribosome with the Mettl15-Mettl17 heterodimer, using the T. brucei structure as a template. We started the modeling by superposing human Mettl17 (PDB ID 8CST) and Mettl15 (PDB ID 7PNX) onto the T. brucei structure with the conserved rRNA core to obtain a model of the heterodimer. In the initial superposition, a short surface-exposed flexible insertion loop of human Mettl17 (residues 232-  

240) clashed with Mettl15 at the interface. It exhibits a variable length and sequence among homologs (Extended Data Fig. 2), suggesting it can adopt an alternative conformation compatible with dimer formation. The clashing loop was rebuilt with AlphaFold (see Methods).  

Next, we aligned the Mettl15-Mettl17 model onto the early assembly stage structure of the human mitoribosome (PDB ID 8CST) using Mettl17 as an anchor (Extended Data Fig. 3). No clashes were observed in this model, and the position of Mettl15 is rotated by $45^{\circ}$ compared to the post-catalytic state. The active site with the cofactor is located more than 40 Å from its target nucleotide C1486. The exact distance could not be calculated, because this rRNA region is disordered in the model. Since the position of Mettl15 is compatible with the human early-stage pre-mitoribosome, we conclude that the modelled intermediate with Mettl15-Mettl17 heterodimer corresponds to a pre-catalytic state (Extended Data Fig. 3).  

# Molecular dynamics simulations suggest how Mettl15 recognizes C1486  

Since neither reported structures nor our models produced a catalytic state, where Mettl15 would be bound to C1486, we used molecular dynamics simulations to gain insight into the conformational motions that would be required for Mettl15 to reach a catalytically compatible state. Specifically, we asked whether Mettl15-bound SAM is able to closely approach C1486 while Mettl15 maintains its post-catalytic specific interactions with the mtSSU (based on the available post-catalytic state). For this purpose, we used an all-atom structure-based (SMOG$^{47}$) force field, where the post-catalytic structure (PDB ID 7PNX$^{22}$) is explicitly defined to be the global potential energy minimum. Structure-based force fields are well-suited to investigate low-energy motions (i.e. accessible via thermal energy) since they provide predictions of molecular flexibility that are consistent with experimental B-factors$^{48}$ and more detailed explicit-solvent simulations$^{49}$. This has allowed these force fields to be used to characterize molecular flexibility and large-scale conformational rearrangements in mitochondrial and cytosolic ribosomes$^{4,50}$. In the current simulations, we define the rRNA residues proximal to C1483 to be disordered (i.e. residues U1477 to C1494 and A1555 to G1570; see methods), since they are unresolved or have large B-factors in the pre-catalytic state (PDB ID 8CST$^{26}$). Since all non-hydrogen atoms are included in this model, these simulations indicate structural fluctuations that arise from thermal energy are sufficient for Mettl15-bound SAM to closely approach C1486, which is a minimal requirement for catalysis to occur (Fig. 4A).  

Our simulations indicate that large-scale structural deformations in Mettl15 are not required for Mettl15-bound SAM and C1486 to adopt close conformations. To demonstrate this, we  

introduced a restraint between atom C41 of C1486 and the sulfur atom of the Mettl15-bound SAM molecule. In the simulation, an apparent rotation of Mettl15 is associated with the adoption of short distances (\~6.5 Å) between C1486 and SAM (Fig. 4A). These rotated conformations of Mettl15 are associated with small scale bending motions of residues Lys271 to His279. To further characterize these deformations, we performed a second set of simulations in which the restraint was not included. We then compared the average spatial deviation of each residue in Mettl15 after alignment to a post-catalytic structure (Fig. 4B). The most significant difference between the restrained and unrestrained simulations was found for residue Leu274, where the average spatial deviation (a.s.d) value increased only slightly, from \~2.1 Å to \~2.7 Å.  

We also find that short distances between SAM and C1486 can arise through low energy rotational motion of Mettl15. To describe the apparent rotational motion that was present when a restraint was included (above), we calculated the rotation ($\gamma$) and tilting ($\theta$) angles (Fig. 4C see methods). The rotation angle ($\gamma$) was defined as rotation that is parallel to that observed between the pre- and post-catalytic structures. In addition, the tilting angle ($\theta$) is defined as rotation that is orthogonal to $\gamma$. In simulations that included the C1486-SAM restraint, we calculated the rotation and tilt angle for all conformations in which the SAM-C1486 distance was less than 7 Å. This revealed that many of these conformations were associated with low rotation angles ($|\gamma| < 3^{\circ}$) and larger tilting angles (7-12$^{\circ}$). To probe the energetics of these tilted conformations, we used our unrestrained simulations to calculate the free energy as a function of tilt angle. This showed that tilt angles of 7-12$^{\circ}$ are only associated with an increase in free energy of \~ 1-5 k$_{\text{B}}$T, relative to the post-catalytic structure (Fig. 4D). This indicates that thermally-induced structural fluctuations about a post-like orientation are sufficient for Mettl15 to position SAM within the vicinity of C1486.  

# Sequential steps of small mitoribosomal subunit assembly involving Mettl17 and Mettl15  

To establish the molecular sequence of Mettl17 and Mettl15 function on the pre-mitoribosome, we ordered the previously obtained structural insights into a series (Fig. 5, Supplementary Video 1). First, the model from the early assembly stage with TFB1M, along with the T. brucei-based model of the Mettl15-Mettl17 heterodimer represents a pre-catalytic state. The next state obtained from molecular dynamics simulations, involves a rearrangement of Mettl15 with a $45^{\circ}$ rotation, bringing SAM within 7 Å from the target to provide substrate for its methylation. Since RbfA, and not TFB1M, is present in the model, it is possible that the association of RbfA and the dissociation of TFB1M lead to a disruption of contacts between Mettl15 and Mettl17 resulting in the departure of Mettl17 from the pre-mtSSU. Therefore, only upon the release of Mettl17 can Mettl15 rotate towards C1486 to induce methylation. This sequence of events provides Mettl15 with the conformational space to approach its rRNA target site as predicted by the simulations (Fig. 4). Finally, when methylation is accomplished, the conformation of Mettl15 changes again with a backward rotation to adopt a loosely bound state with SAH being 45 Å away from the target. This would ultimately lead to the replacement of Mettl15 by initiation factors in the late stage marking the completion of the mtSSU assembly as previously reported$^{22}$.  

This architecture defines Mettl17 as the key factor that structurally orchestrates the series of assembly events. On one hand, its presence allows the binding of the methyltransferase Mettl15 required for rRNA maturation, and on the other hand, its departure provides the conformational potential of Mettl15 central domains facilitating the rRNA maturation. Thus, Mettl17 stimulates the modification without exhibiting enzymatic activity.  

# Discussion  

In this analysis, we present in silico model of human pre-mitoribosomal assembly, revealing that coupled methyltransferases Mettl15 and Mettl17 are involved in previously undetected, transient assembly states. Our findings indicate that Mettl17 functions as a recruitment factor for Mettl15, forming a structural checkpoint for early assembly stages. This association suggests a broader quality control mechanism where Mettl17, alongside TFB1M, stabilizes Mettl15 and pauses maturation. Release of Mettl17 then facilitates Mettl15's conformational change on pre-mitoribosome, allowing catalytic methylation of the rRNA, which aligns with observations of the folded rRNA region in this pre-mtSSU assembly$^{22}$. The precise mtSSU head position during C1486 methylation could differ, since it is rotated between pre- and post  

catalytic state by 15°. Upon completion of the methylation, Mettl15 is released, and the subunit core can move toward its functional conformation (Fig. 6). Thus, our integrative structural analysis not only suggests a more complete picture of the mechanistic assembly, but also provides an experimentally-testable hypothesis regarding a potential quality-control mechanism.  

These steps of mitoribosomal assembly are particularly important in the context of biochemical, physiological, and behavioural observations in animals lacking Mettl15$^{51}$. In mice, the loss/ablation/downregulation of Mettl15 has been shown to lead to suboptimal muscle performance, decreased learning capabilities, and lower blood glucose level after physical exercise$^{51}$. The same study, as well as results obtained earlier for cell cultures$^{36}$ also reported accumulation of the RbfA factor, and our model is consistent with these data.  

SAM is essential for RNA processing in mouse embryonic fibroblasts and skeletal muscle$^{52}$. Although Mettl17 retains the features typical of class I SAM-dependent methyltransferases$^{40}$, it does not methylate the 12S rRNA region, despite coming into contact with it during assembly. Our models suggesting that Mettl17 acts in recruitment of Mettl15 explain why loss of Mettl17 leads to around 70% reduction in the methylation, resulting in the impaired translation of mitochondrial protein-coding genes and consequent changes in the cellular metabolome$^{39}$. Therefore, it appears that Mettl17 acts as an enhancer of the mtSSU rRNA stability without  

being involved in RNA modification. This provides a more complete description of mtSSU assembly and proposes a plausible explanation for the sequential maturation of the human mitoribosome. Because the two methyltransferases co-exist in most eukaryotes, the described functional coupling most likely predates the last common eukaryotic ancestor, and its function presumably became vital as a consequence of the evolution of mitochondrial ribosomes during eukaryogenesis.  

Finally, our methodology shows how integrating molecular dynamics with template-based modelling can reveal steps missed in experimental captures due to their transient nature. Although this study has limitations that require further experimental validation, the combined methodology presented here may serve as a more general complementary approach for revealing missing mechanistic steps of transient associations. Together with automated workflows for model building$^{53,54}$, that further integrate diffusion models AF2-predicted structures$^{55}$, scaled up by deep learning systems that generate protein ensembles$^{56}$, this approach can be used for exploring dynamic properties of complex macromolecular systems where only partial experimental data is available. Our work underlines the importance of studying intricate biological processes in combination with advanced computational analyses in order to ultimately predict protein function and derive biogenesis pathways.  
| Protein | Previous name or chain ID | Newly described feature(s) |
| --- | --- | --- |
| mt-SAF38 | chains UY, Ue | newly identified assembly factor |
| Mettl15 | mt-SAF14 | homolog of Mettl15 (RsmH in E. coli) cofactor SAM |
| Mettl17 | mt-SAF1 | homolog of Mettl17 cofactor SAM iron-sulphur cluster Fe4S4 |
| RbfA | mt-SAF18 | homolog of RbfA |
| mt-SAF16 mt-SAF19 mt-SAF25 | - | homologs of Saccharomyces cerevisiae Mam3357 (Uniprot ID P40513), human p32 (Q0702158) and Chlamydomonas reinhardtii mtSSU protein mS10559 (A0A2K3DAY3) |
| mS53 | - | residues 63-84 modeled |
| mt-SAF5 | - | residues 560-596 modeled |
| mt-SAF10 | - | residues 4-6 modeled |
| mt-SAF11 | - | residues 148-156 modeled |
| rRNA | - | several regions modeled or adjusted (see Methods) |
| mt-SAF10 mt-SAF22 mt-SAF27 | chains UB, UC, UD, UF, UI, UJ, UM, UN | ligand acetyl coenzyme A |  

# METHODS  

# Model building  

The PDB ID 6SGB$^{15}$ was used as a starting point and modified as follows. A new assembly factor, mt-SAF38 (Tb927.5.1720), was assigned to an unknown chain in the original model based on the local density (cryoEM density map EMD-10180) and the presence of the protein in previously isolated trypanosomal mitoribosomal complexes$^{18}$ as revealed by mass spectrometry. Several regions were added or extended in different proteins. In the protein mS53 (chain DF), residues 63-84 were included. N-terminal regions were extended in the models of the proteins mt-SAF10 (chain FA) and mt-SAF11 (chain FB) protein. In the protein mt-SAF5 (chain F5), residues 560-596, previously categorized as an unknown chain, was now modeled. Several proteins have been identified as homologs of assembly factors from other organisms: mt-SAF1 has been assigned as Mettl17, mt-SAF14 as Mettl15, and mt-SAF18 as RbfA. Structural similarity revealed three components of the heterotrimeric assembly mt-SAF16, mt-SAF19, and mt-SAF25 are homologs of the homotrimer-forming human protein p32, yeast Mam33, or algal mS105.  

The model of rRNA was modified as follows. The linker between nucleotides 560-620 (h44, h45) was adjusted. Some regions with insufficient resolution quality of density were removed, namely nucleotides 208-226, 254-260, 349-353, 385-389, 397-417, 431-440, 489-510, and 523-529. Nucleotides 67-73, 80-87, 171-172, 183-189, 273-285, 322-324, and 366-374 were shown as ribose-phosphate backbones.  

Several ligands were included in the model. Consistent with previous observations$^{23}$, we identified the density corresponding to GTP in the protein mS29 and PO$_4^{3-}$ in the protein mt-SAF29. Ligand in Mettl17 and Mettl15, originally assigned as S-adenosylhomocysteine (SAH) molecules were substituted with S-adenosylmethionines (SAM), because there is no evidence suggesting that these proteins exist in the post-catalytic state, and the presence of SAM is more plausible in the context of our results. Furthermore, the Zn$^{2+}$ ion present in Mettl17 was replaced with an Fe$_4$S$_4$ iron-sulfur cluster, consistently with the density, identity of coordinating residues and recent identification of iron-sulfur cluster in yeast and mammalian homologs. Unidentified chain UD was replaced with acetyl Co-A.  

# Structure prediction and analyses  

Structure prediction was performed by AlphaFold3$^{30}$ or AlphaFold Multimer$^{28}$. The latter was used with databases BFD, Mgnify 2018_12, UniRef30 2021_03, UniRef90 2023_04 to predict structures and calculate ipTM scores for Mettl15 in dimer with all assembly factors present in early mtSSU intermediate. For T. brucei homology model, clashes between Mettl15 and RNA were fixed by inpainting as described in AF_unmasked$^{62}$. Here, the Mettl15-Mettl17 complex was used as a multimeric template and the clashing loop was deleted from the template so that it could be rebuilt by AlphaFold. Fifty predictions were generated this way, and the one closest to the initial template (RMSD: 0.2) where the clash would be fixed when including the RNA was selected. We neither show nor interpret regions with pLDDT scores below 65 in any of the models. Angles between Mettl15 in different models were calculated using the PyMOL (Schrödinger, US)$^{63}$ built-in script angle_between_domains.  

Identification and phylogenetic analyses of Mettl15 and Mettl17 across eukaryotes

Using Escherichia coli, Homo sapiens, and Trypanosoma brucei orthologs of Mettl15 and Mettl17 as queries for blastp search against the EukProt v3 database$^{64}$, we built starting datasets that were subsequently cleaned from apparent eukaryotic contaminations using  

phylogenetically-aware approach (identification of possible contaminants by visual inspection of phylogenetic tree followed by manual check of their origin). Cleaned datasets were used to build profiles hmm in HMMER3$^{65}$. Next, 131 organisms that cover known eukaryotic diversity and whose genome or transcriptome assemblies are of a good quality were selected for the final search. This search was performed in three steps: 1/ HMMER3 search with profiles hmm; 2/ blastp search$^{66}$ using query sequence from a closely related species; 3/ tblastn search in corresponding nucleotide assembly (to exclude possibility that ortholog is missing due to an inaccurate protein prediction). Names of selected organisms, accession numbers of used assemblies, and tools that were used for successful search are indicated in the Supplementary Table 2. Multiple sequence alignments of the homologous amino acid sequences were built using MAFFT v7.407 with the L-INS-i algorithm$^{67}$ and were manually trimmed to exclude unreliably aligned regions. The maximum likelihood tree was inferred with IQ-TREE multicore v2.2.0.3$^{68}$ using the LG4X substitution model. Statistical support was assessed with IQ-TREE non-parametric bootstrap replicates. Sequences of both genes from all organisms are available in Supplementary Data 1&2.  

# Molecular dynamics simulations  

# Potential energy function  

An all-atom structure-based "SMOG" model$^{46}$ of the mitoribosome small subunit was used to probe the scale of structural fluctuations around the post-catalytic state and determine whether thermal energy is sufficient for Mettl15-bound SAM to approach C1486, or whether Mettl15 is more likely to be associated with a larger-scale rearrangement that would require transient dissociation from the ribosome. The force field that was used is a single-basin model where the post-catalytic structure (PDB ID 7PNX) was defined as the global potential energy minimum. The specific variant of the force field is available through the smog-server force field repository (https://smog-server.org), with entry name AA_PTM_Hassan21.v2. The functional form of the potential energy is given as:  

$$

\begin{array} { r l } { U = \displaystyle \sum _ { \mathrm { b o n d s } } \frac { \epsilon _ { r } } { 2 } \left( r _ { i } - r _ { i , 0 } \right) ^ { 2 } + \displaystyle \sum _ { \mathrm { a n g l e s } } \frac { \epsilon _ { \theta } } { 2 } \left( \theta _ { i } - \theta _ { i , 0 } \right) ^ { 2 } + \displaystyle \sum _ { \mathrm { i m p r o p e r s } } \frac { \epsilon _ { \chi _ { \mathrm { i m p } } } } { 2 } \left( \chi _ { i } - \chi _ { i , 0 } \right) ^ { 2 } } & { } \\ { + \displaystyle \sum _ { \mathrm { p l a n a r } } \epsilon _ { \mathrm { p l a n a r } } [ 1 - \cos ( 2 \chi _ { i } ) ] + \displaystyle \sum _ { \mathrm { b a c k p r o e n ~ d i h e d r a l s } } \epsilon _ { \mathrm { b b } } \, F \big ( \phi _ { i } - \phi _ { i , 0 } \big ) } & { } \\ { + \displaystyle \sum _ { \mathrm { s i d e c h a i n ~ d i h e d r a l s } } \epsilon _ { \mathrm { s c } } \, F \big ( \phi _ { i } - \phi _ { i , 0 } \big ) + \displaystyle \sum _ { \mathrm { c o n t a c t s } } \epsilon _ { c } \left[ \left( \frac { \sigma _ { i j } } { r _ { i j } } \right) ^ { 1 2 } - 2 \left( \frac { \sigma _ { i j } } { r _ { i j } } \right) ^ { 6 } \right] } & { } \\ { + \displaystyle \sum _ { \mathrm { n o n - c o n t a c t s } } \epsilon _ { \mathrm { n c } } \left( \frac { \sigma _ { \mathrm { n c } } } { \sigma _ { i j } } \right) ^ { 1 2 } } & { } \end{array}

$$  

where  

$$

F ( \phi ) = [ 1 - c o s ( \phi ) ] + \frac { 1 } { 2 } [ 1 - c o s ( 3 \phi ) ]

$$  

$\{r_{0}\}$ and $\{\theta_{0}\}$ parameters are given values found in the Amber ff03 force field$^{69}$. Dihedral parameters $\{\chi_{0}\}$ and $\{\phi_{0}\}$ are assigned the corresponding values found in the experimental model. Non-bonded contacts that are found in the experimental model, are identified according to the Shadow Contact Map algorithm, with a shadowing radius of 1 Å and a cutoff distance of 6 Å. The contacts are given an attractive 6-12 interaction that stabilizes the preassigned structure, with interatomic distance $\sigma_{ij}$ that is found in the experimental structure, multiplied by 0.96 to avoid artificial expansion of the structure$^{70-72}$. Atom pairs that are not in contact are assigned a repulsive potential to model excluded-volume steric interactions, where $\sigma_{nc}$ is given the value 2.5 Å. Energy scale weights are defined as $\epsilon_{r}=100\ \frac{\epsilon}{\AA^{2}}$, $\epsilon_{\theta}=80\ \frac{\epsilon}{rad^{2}}$, $\epsilon_{\chi_{imp}}=10\ \frac{\epsilon}{rad^{2}}$, $\epsilon_{\chi_{planar}}=40\ \frac{\epsilon}{rad^{2}}$, $\epsilon_{nc}=0.1\ \epsilon$,where $\epsilon$ is the reduced energy unit. The dihedral and contact energy weights are normalized as in Whitford et al, Proteins 2009.  

Since rRNA residues near the Mettl15 binding site are disordered (unresolved or high B-factors) in the pre-catalytic structure, these regions were modelled as disordered. For this, stabilizing contacts and dihedrals for the flexible rRNA region (i.e. residues U1477 to C1494 and A1555 to G1570), were removed.  

Two sets of simulations were performed. In the first set of simulations, a harmonic restraint was introduced, which ensured that the distance between C1486 and SAM (atom name) adopted short values. The harmonic restraint had a minimum at 5 Å and the spring constant was $150 \frac{\epsilon}{nm^2}$. These simulations were used to ask whether simple bending motions of Mettl5 are sufficient for SAM and C1486 to become proximal. In the second set of simulations, the  

restraint was not included. These unrestrained simulations were performed to determine the scale and direction of structural fluctuations that can arise from thermal energy.  

# Simulation details  

All force field files were generated using SMOG2 software package$^{47}$. Molecular dynamics simulations were performed using OpenMM$^{50}$ and OpenSMOG$^{71}$ libraries. The simulations were performed at a reduced temperature of $0.5\frac{\epsilon}{k_{B}}$ that was maintained by using Langevin dynamics protocols.  

# Calculating rotation angles for Mettl5  

Euler angles were used to describe rotation of Mettl15, relative to the mtSSU body$^{69,70}$. Consistent with methods for describing rotation of the mtSSU$^{69}$, we described a rotation angle $\gamma$, which is the sum of the $\psi$ and $\phi$ angles in the Euler formulation (Extended Data Figure 4). The polar angle $\theta$ (i.e. tilt angle) represents rotation that is orthogonal to the primary rotation. To calculate Euler angles, we first assigned a set of axes that remain fixed in the frame of reference of Mettl15. For convenience, we define the "Z" axis as the axis of rotation (Euler-Rodrigues axis) between the pre and post catalytic states. The following protocol was used to define the primary rotation axis:  

1. Least squares alignment of the mtSSU (excluding Mettl15) of the pre-catalytic structure to post-catalytic structure.  

2. Associate coordinate system to Mettl15 of pre and post catalytic structures.  

3. The E-R angle was then calculated between the coordinate systems of both structures. The angle was found to be $45^{\circ}$.  

To calculate Euler angles in the simulations, the following protocol was applied:  

1. Define "Z" axis as the E-R axis. This ensures that our primary rotation angle $\gamma$ describes rotation that is parallel to that defined by the pre-to-post rearrangement.  

2. Align each simulated frame to the post-catalytic structure of the mtSSU, where alignment was based on non-Mettl15 atoms.  

3. Align the post-catalytic conformation of Mettl15 to each simulated frame.  

4. Calculate the Euler angles ($\phi$, $\psi$ and $\theta$) between the post-catalytic and aligned (previous step) orientation.  

5. Define rotation as $\gamma = \phi + \psi$.  

6. Define tilt as $\theta$.  

# Source data  

The atomic model of the $T$. brucei mtSSU precursor was deposited in the PDB database (PDB ID 9HNY). All data from phylogenetic and structural analyses are available as supplementary material or have been deposited on Figshare (link will be provided in the accepted version).  

# Acknowledgements  

This work was supported by the European Research Council (ERC-2018-StG-805230), Czech Science Foundation (20-04150Y) to O.G., the project P JAC CZ.02.01.01/00/22_008/0004575 RNA for therapy, co-funded by the European Union, and the Ministry of Education, Youth and Sports of the Czech Republic through the e-INFRA CZ (ID:90254) to O.G. and A.Z., and SciLifeLab BeyondFold to B.N. G.W. and P.C.W were supported by NIH grant R35GM153502-01. Some of the structure prediction experiments and other analyses were enabled by the Berzelius resource provided by the Knut and Alice Wallenberg Foundation at the National Supercomputer Centre in Sweden. Work in the Center for Theoretical Biological Physics was supported by the National Science Foundation (NSF) grant PHY-2210291. We thank S. Aibara, Y. Itoh, V. Singh, and V. Tobiasson for their contributions to model building, data interpretation, and discussions.  

# Author contributions  

Y.Z. built the model; C.M. carried out computational modeling and structure prediction; G.W. and P.C.W. performed molecular simulations; Y.Z., C.M., G.W., P.C., P.C.W., O.G., A.A. performed the structural analysis; T.P. performed the phylogenetic analysis; B.N., A.Z. supervised the project; A.A. and O.G. wrote the manuscript with help from Y.Z., P.C.W. All the authors contributed to the manuscript preparation.  

Extended Data Fig. 1. New features in the map. (A) The density and model of the newly identified mt-SAF38 with close-up views showing how residues fit the density. Right, mt-SAF38 (dark red) superposed with mouse acyl-coA thioesterase (PDB 5ZV3). (B) Acetyl-CoA placed into a density associated with tryptophan 25 of mt-SAF22 and other examples of the hammerhead shaped densities. Sequence logo of acetyl-CoA binding regions, showing the conserved tryptophan, was created using WebLogo$^{73}$.  

Extended Data Figure 2: Flexibility and conservation of Mettl17. (A) PDB ID 8CST colored by B-factor. Increased flexibility of the loop 232-240 is evident by a higher B-factor. (B) The conservation coloring profile calculated by ConSurf repository$^{74}$ mapped onto the model.  

Extended Data Figure 3: Comparison of the Mettl15-Mettl17 heterodimer in T. brucei and corresponding in silico model of the human mitoribosome. Human early-stage pre-mitoribosomal model PDB ID 8CST was aligned onto T. brucei Mettl17, and Mettl15 was modelled based on the trypanosomal template with no clashes. The position of Mettl15 in the created in silico model is compatible with the experimental data.  

Extended Data Figure 4: Description of Euler angles used to analyze molecular dynamics simulation. The angles were calculated by comparing the vectors of the model plane $z^{'}, x^{' }$ with the corresponding vectors of the reference plane $z^{-}, x^{-}$. Rotation is defined be angle $\gamma = \phi + \psi$, while tilt is defined by angle $\theta$ around the tilting axis (line of nodes).  

# References  

1. Singh, V. et al. Structural basis of LRPPRC-SLIRP-dependent translation by the mitoribosome. Nat Struct Mol Biol (2024).

2. Itoh, Y. et al. Mechanism of membrane-tethered mitochondrial protein synthesis. Science 371, 846-849 (2021).

3. Ott, M., Amunts, A. & Brown, A. Organization and Regulation of Mitochondrial Protein Synthesis. Annu Rev Biochem 85, 77-101 (2016).

4. Singh, V. et al. Mitoribosome structure with cofactors and modifications reveals mechanism of ligand binding and interactions with L1 stalk. Nat Commun 15, 4272 (2024).

5. Amunts, A., Brown, A., Toots, J., Scheres, S.H.W. & Ramakrishnan, V. Ribosome. The structure of the human mitochondrial ribosome. Science 348, 95-98 (2015).

6. Greber, B.J. et al. Ribosome. The complete structure of the 55S mammalian mitochondrial ribosome. Science 348, 303-8 (2015).

7. Lavdovskaia, E. et al. A roadmap for ribosome assembly in human mitochondria. Nat Struct Mol Biol (2024).

8. Conor Moran, J., Del'Olio, S., Choi, A., Zhong, H. & Barrientos, A. Mitoribosome Biogenesis. Methods Mol Biol 2661, 23-51 (2023).

9. Brischigliaro, M., Sierra-Magro, A., Ahn, A. & Barrientos, A. Mitochondrial ribosom biogenesis and redox sensing. FEBS Open Bio 14, 1640-1655 (2024).

10. Khawaja, A., Cipullo, M., Kruger, A. & Rorbach, J. Insights into mitoribosomal biogenesis from recent structural studies. Trends Biochem Sci (2023).

11. Haas, R.H. Mitochondrial Dysfunction in Aging and Diseases of Aging. Biology (Basel) 8(2019).

12. Hong, H.J. et al. Mitoribosome insufficiency in beta cells is associated with type 2 diabetes-like islet failure. Exp Mol Med 54, 932-945 (2022).

13. Richman, T.R. et al. Mitochondrial mistranslation modulated by metabolic stress causes cardiovascular disease and reduced lifespan. Aging Cell 20, e13408 (2021).  

14. Pecina, P. et al. Haplotype variability in mitochondrial rRNA predisposes to metabolic syndrome. Commun Biol 7, 1116 (2024).

15. Sauer, M. et al. Mitoribosomal small subunit biogenesis in trypanosomes involves an extensive assembly machinery. Science 365, 1144-1149 (2019).

16. Soufari, H. et al. Structure of the mature kinetoplastids mitoribosome and insights into its large subunit biogenesis. Proc Natl Acad Sci U S A 117, 29851-29861 (2020).

17. Jaskolowski, M. et al. Structural Insights into the Mechanism of Mitoribosomal Large Subunit Biogenesis. Mol Cell 79, 629-644 e4 (2020).

18. Tobiasson, V. et al. Interconnected assembly factors regulate the biogenesis of mitoribosomal large subunit. EMBO J 40, e106292 (2021).

19. Lenarcic, T. et al. Mitoribosomal small subunit maturation involves formation of initiation-like complexes. Proc Natl Acad Sci U S A 119(2022).

20. Kummer, E. et al. Unique features of mammalian mitochondrial translation initiation revealed by cryo-EM. Nature 560, 263-267 (2018).

21. Khawaja, A. et al. Distinct pre-initiation steps in human mitochondrial translation. Nat Commun 11, 2932 (2020).

22. Itoh, Y. et al. Mechanism of mitoribosomal small subunit biogenesis and preinitiation. Nature (2022).

23. Itoh, Y. et al. Structure of the mitoribosomal small subunit with streptomycin reveals Fe-S clusters and physiological molecules. Elife 11(2022).

24. Aibara, S., Singh, V., Modelska, A. & Amunts, A. Structural basis of mitochondrial translation. Elife 9(2020).

25. Pham, T.C.P. et al. The mitochondrial mRNA-stabilizing protein SLIRP regulates skeletal muscle mitochondrial structure and respiration by exercise-recoverable mechanisms. Nat Commun 15, 9826 (2024).

26. Harper, N.J., Burnside, C. & Klinge, S. Principles of mitoribosomal small subunit assembly in eukaryotes. Nature 614, 175-181 (2023).

27. Ast, T. et al. METTL17 is an Fe-S cluster checkpoint for mitochondrial translation. Mol Cell 84, 359-374 e8 (2024).

28. Evans, R. et al. (2022).

29. Wallner, B., Amunts, A., Naschberger, A., Nystedt, B. & Mirabello, C. (2022).

30. Jumper, J. & Hassabis, D. Protein structure predictions to atomic accuracy with AlphaFold. Nat Methods 19, 11-12 (2022).

31. Schedlbauer, A. et al. A conserved rRNA switch is central to decoding site maturation on the small ribosomal subunit. Sci Adv 7(2021).

32. Bikmullin, A.G. et al. Yet Another Similarity between Mitochondrial and Bacterial Ribosomal Small Subunit Biogenesis Obtained by Structural Characterization of RbfA from S. aureus. International Journal of Molecular Sciences 24(2023).

33. Datta, P.P. et al. Structural aspects of RbfA action during small ribosomal subunit assembly. Mol Cell 28, 434-45 (2007).

34. Rozanska, A. et al. The human RNA-binding protein RBFA promotes the maturation of the mitochondrial ribosome. Biochem J 474, 2145-2158 (2017).

35. Van Haute, L. et al. METTL15 introduces N4-methylcytidine into human mitochondrial 12S rRNA and is required for mitoribosome biogenesis. Nucleic Acids Res 47, 10267-10281 (2019).

36. Laptev, I. et al. METTL15 interacts with the assembly intermediate of murine mitochondrial small ribosomal subunit to form m4C840 12S rRNA residue. Nucleic Acids Res 48, 8022-8034 (2020).

37. Mutti, C.D., Van Haute, L. & Minczuk, M. The catalytic activity of methyltransferase METTL15 is dispensable for its role in mitochondrial ribosome biogenesis. RNA Biol 21, 23-30 (2024).

38. Chen, H. et al. The human mitochondrial 12S rRNA m(4)C methyltransferase METTL15 is required for mitochondrial function. J Biol Chem 295, 8505-8513 (2020).

39. Shi, Z. et al. Mettl17, a regulator of mitochondrial ribosomal RNA modifications, is required for the translation of mitochondrial coding genes. FASEB J 33, 13040-13050 (2019).

40. Mashkovskaia, A.V. et al. Testing a Hypothesis of 12S rRNA Methylation by Putative METTL17 Methyltransferase. Acta Natura 15, 75-82 (2023).

41. Zhong, H. et al. BOLA3 and NFU1 link mitoribosome iron-sulfur cluster assembly to multiple mitochondrial dysfunctions syndrome. Nucleic Acids Res 51, 11797-11812 (2023).

42. Gahura, O., Chauhan, P. & Zikova, A. Mechanisms and players of mitoribosomal biogenesis revealed in trypanosomatids. Trends Parasitol 38, 1053-1067 (2022).

43. Tyc, J., Novotna, L., Pena-Diaz, P., Maslov, D.A. & Lukes, J. RSM22, mtYsxC and PNKD-like proteins are required for mitochondrial translation in Trypanosoma brucei. Mitochondrion 34, 67-74 (2017).

44. Dimmer, K.S. et al. Genetic basis of mitochondrial function and morphology in Saccharomyces cerevisiae. Mol Biol Cell 13, 847-53 (2002).

45. Petrov, A.S. et al. Structural Patching Fosters Divergence of Mitochondrial Ribosomes. Mol Biol Evol 36, 207-219 (2019).

46. Zhu, W., Shenoy, A., Kundrotas, P. & Elofsson, A. Evaluation of AlphaFold-Multimer prediction on multi-chain protein complexes. Bioinformatics 39(2023).

47. Noel, J.K. et al. SMOG 2: A Versatile Software Package for Generating Structure-Based Models. PLoS Comput Biol 12, e1004794 (2016).

48. Whitford, P.C. et al. Accommodation of aminoacyl-tRNA into the ribosome involves reversible excursions along multiple pathways. RNA 16, 1196-204 (2010).

49. Jackson, J., Nguyen, K. & Whitford, P.C. Exploring the balance between folding and functional dynamics in proteins and RNA. Int J Mol Sci 16, 6868-89 (2015).

50. Freitas, F.C., Fuchs, G., de Oliveira, R.J. & Whitford, P.C. The dynamics of subunit rotation in a eukaryotic ribosome. Biophysica 1, 204-221 (2021).

51. Averina, O.A. et al. Mitochondrial rRNA Methylation by Mettl15 Contributes to the Exercise and Learning Capability in Mice. Int J Mol Sci 23(2022).

52. Glasgow, R.I.C. et al. The mitochondrial methylation potential gates mitoribosome assembly. in bioRxiv (2024).

53. Jamali, K. et al. Automated model building and protein identification in cryo-EM maps. Nature 628, 450-457 (2024).

54. Su, B., Huang, K., Peng, Z., Amunts, A. & Yang, J. Improved automated model building for cryo-EM maps using CryFold. in bioRxiv (2024).

55. Wang, X., Zhu, H., Terashi, G., Taluja, M. & Kihara, D. DiffModeler: large macromolecular structure modeling for cryo-EM maps using a diffusion model. Nat Methods 21, 2307-2317 (2024).

56. Lewis, S. et al. (2024).

57. Pu, Y.G. et al. Crystal structures and putative interface of Saccharomyces cerevisiae mitochondrial matrix proteins Mmf1 and Mam33. J Struct Biol 175, 469-74 (2011).

58. Jiang, J., Zhang, Y., Krainer, A.R. & Xu, R.M. Crystal structure of human p32, a doughnut-shaped acidic mitochondrial matrix protein. Proc Natl Acad Sci U S A 96, 3572-7 (1999).  

59. Waltz, F. et al. How to build a ribosome from RNA fragments in Chlamydomonas mitochondria. Nat Commun 12, 7176 (2021).

60. Burki, F., Roger, A.J., Brown, M.W. & Simpson, A.G.B. The New Tree of Eukaryotes. Trends Ecol Evol 35, 43-55 (2020).

61. Krissinel, E. & Henrick, K. Inference of macromolecular assemblies from crystalline state. J Mol Biol 372, 774-97 (2007).

62. Mirabello, C., Wallner, B., Nystedt, B., Azinas, S., & Carroni, M. Unmasking AlphaFold to integrate experiments and predictions in multimeric complexes. Nature Communications, 15(1), 8724 (2024).

63. Schrödinger, L. & DeLano, W., 2020. PyMOL, Available at: http://www.pymol.org/pymol.

64. Richter, D.J. et al. EukProt: A database of genome-scale predicted proteins across the diversity of eukaryotes. Peer Community Journal 2(2022).

65. Mistry, J., Finn, R.D., Eddy, S.R., Bateman, A. & Punta, M. Challenges in homology search: HMMER3 and convergent evolution of coiled-coil regions. Nucleic Acids Res 41, e121 (2013).

66. Altschul, S.F. et al. Gapped BLAST and PSI-BLAST: a new generation of protein database search programs. Nucleic Acids Res 25, 3389-402 (1997).

67. Katoh, K. & Standley, D.M. MAFFT multiple sequence alignment software version 7: improvements in performance and usability. Mol Biol Evol 30, 772-80 (2013).

68. Nguyen, L.T., Schmidt, H.A., von Haeseler, A. & Minh, B.Q. IQ-TREE: a fast and effective stochastic algorithm for estimating maximum-likelihood phylogenies. Mol Biol Evol 32, 268-74 (2015).

69. Hassan, A. et al. Ratchet, swivel, tilt and roll: a complete description of subunit rotation in the ribosome. Nucleic Acids Res 51, 919-934 (2023).

70. Nguyen, K. & Whitford, P.C. Steric interactions lead to collective tilting motion in the ribosome during mRNA-tRNA translocation. Nat Commun 7, 10586 (2016).

71. de Oliveira, A.B., Jr. et al. SMOG 2 and OpenSMOG: Extending the limits of structure-based models. Protein Sci 31, 158-172 (2022).

72. Eastman, P. et al. OpenMM 8: Molecular Dynamics Simulation with Machine Learning Potentials. J Phys Chem B 128, 109-116 (2024).

73. Crooks, G.E., Hon, G., Chandonia, J.M. & Brenner, S.E. WebLogo: a sequence logo generator. Genome Res 14, 1188-90 (2004).

74. Yariv, B. et al. Using evolutionary data to make sense of macromolecules with a "face-lifted" ConSurf. Protein Sci 32, e4582 (2023).  
