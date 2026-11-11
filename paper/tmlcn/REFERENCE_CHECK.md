# Reference check (57 cited entries)

Every cited entry of `references.bib` was checked against a source outside this repository.
Publisher and proceedings sites could not be opened from the build machine, so two kinds of
source were used and are distinguished below:

- **[D]** a record file that was downloaded and read: PMLR source records
  (`mlresearch/vNNN`), DBLP records as redistributed in `yuchenlin/rebiber`, CryptoBib
  (`cryptobib/export`), and the authors' own repositories;
- **[S]** the content of the named publisher, index or library page as returned by a web search.

Nothing was filled in from memory. A field that could not be confirmed was removed, not kept.

## Corrections made in this revision

| Entry | Change | Basis |
|---|---|---|
| `rieke2020` | issue "no. 1" removed, article number 119 added | Nature page and PMC7490367: "npj Digital Medicine 3, Article number: 119 (2020)" [S] |
| `platt1999` | title of the printed chapter ("Probabilities for SV Machines"), editors and pages 61-74 added | MIT Press Direct chapter record, DOI 10.7551/mitpress/1113.003.0008; Google Books contents list (chapter starts on p. 61, the next on p. 75) [S]. The title often cited is that of the 1999 manuscript. |
| `duterrail2022flamby` | surname "Ogier du Terrail" (was rendered "J. O. du Terrail") | official NeurIPS BibTeX reproduced in the authors' README [D] |
| `choquette2023banded` | capital of "(Amplified)" protected | rendering check |
| `geiping2020` | pages 16937-16947 added | ACM DL, NIPS'20 [S] |
| `horvath2021fjord` | pages 12876-12889 added | DBLP record [D] |
| `narayanan2021variegated` | pages 610-625 added | NSF PAR record 10291906, DOI 10.1145/3452296.3472923 [S] |
| `liu2024recordlevel` | pages 303-317 added | CryptoBib, DOI 10.1145/3658644.3670351 [D] |
| `wang2019adacomm` | page range removed | could not be confirmed: the MLSys abstract page gives the volume only and DBLP has no pages |
| `pollard2018`, `yang2023medmnist`, `geiping2020`, `mironov2019sgm` | capitalisation of proper names and of the word after a dash protected | published titles |
| five article-number entries | printed as "Art. no." (IEEE style) instead of a page | `rieke2020`, `strack2014`, `pollard2018`, `yang2023medmnist`, `vancalster2019` |
| all conference entries | `volume` removed, so that proceedings are cited uniformly | consistency |
| entries with more than six authors | first author and "et al." throughout | consistency (`kairouz2021`, `rieke2020`, `dayan2021`, `duterrail2022flamby`, `narayanan2021variegated`) |

## Entries confirmed without change

Volume, issue, pages (or article number), year and author list agree with the source named.

- PMLR records [D]: `mcmahan2017` (54: 1273-1282), `khaled2020` (108: 4519-4529), `karimireddy2020` (119: 5132-5143),
  `collins2021` (139: 2089-2099), `kairouz2021ddg` (139: 5201-5212), `kairouz2021ftrl` (139: 5213-5225),
  `pillutla2022` (162: 17716-17758), `balle2018` (80: 394-403), `vero2023tableak` (202: 35051-35083).
- DBLP records [D]: `alistarh2017qsgd`, `stich2019`, `lin2020postlocal`, `diao2021heterofl`, `mcmahan2018dp`,
  `lee2023fedlama` (8491-8499), `yang2024fedas` (11986-11995), `wang2021coop` (JMLR 22(213): 1-50),
  `choquette2024correlated`, `andrew2021adaptive` (17455-17466), `rogers2016odometer` (1921-1929).
- CryptoBib [D]: `yeom2018` (268-282), `mironov2017` (263-275), `carlini2022lira` (1897-1914).
- Authors' repositories [D]: `liu2022crosssilo` (5925-5940), `yang2023medmnist` (Sci. Data 10, 41).
- Publisher, index or library pages [S]: `wang2020fednova` (7611-7623), `choquette2023banded` (74856-74889),
  `zhu2019dlg` (14774-14784, ACM DL), `chen2018lag` (5050-5060, NeurIPS metadata), `ke2017lightgbm` (3146-3154),
  `lyu2017svt` (PVLDB 10(6): 637-648), `strack2014` (PMID 24804245), `vancalster2019` (BMC Med 17:230),
  `huang2012lte` (225-238), `dong2022gdp` (JRSS-B 84(1): 3-37), `wang2019jsac` (37(6): 1205-1221),
  `lim2020` (22(3): 2031-2063), `chen2021joint` (20(1): 269-283), `sattler2020` (31(9): 3400-3413),
  `razmi2022leo` (11(4): 717-721), `kairouz2021` (14(1-2): 1-210), `detrano1989` (64(5): 304-310),
  `pollard2018` (Sci. Data 5, 180178), `dayan2021` (27(10): 1735-1743), `malan2024alf` (11(4): 6072-6083),
  `li2020fedprox` (429-450), `tr38913` (V17.0.0, March 2022).

## Known disagreements between indexes (bib follows the source named first)

- `chen2018lag`: 5050-5060 in the NeurIPS metadata; ACM DL and DBLP give 5055-5065.
- `zhu2019dlg`: 14774-14784 in ACM DL; DBLP gives 14747-14756.
- `ke2017lightgbm`: 3146-3154 in the NeurIPS metadata and DBLP; the authors' README gives 3149-3157.
- `balle2018`: 394-403 in PMLR; DBLP gives 403-412.
- `rogers2016odometer`: author order as in the paper's byline; DBLP lists the authors in another order.

## Not verified directly

No publisher page was opened from the build machine; every [S] entry rests on the search
rendition of the page named. Before the camera-ready version the [S] entries should be
re-read on the publishers' sites.
