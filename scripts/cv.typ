// Typst port of Yuan's resume template (https://github.com/xyz-yuanhf/yuan-resume)
// Compile with: typst compile scripts/cv.typ assets/cv.pdf (deploy.sh does this; only uses fonts bundled with Typst)

// -------------------- STYLE --------------------
#let title-width = 2.7cm
#let content-width = 13.85cm

// Fonts: Libertinus Serif ships with Typst, so no external font files are needed.
// (Originally: Sabon for body text, Calluna for the header, Cronos Pro for section titles.)
#let body-font = "Libertinus Serif"
#let header-font = "Libertinus Serif"
#let title-font = "Libertinus Serif"

#set document(title: "Pablo Miralles - CV", author: "Pablo Miralles")
#set page(
  paper: "a4",
  margin: (top: 1.2cm, bottom: 2.6cm, left: 1.8cm, right: 2.3cm),
  numbering: "1",
)
#set text(font: body-font, size: 10.5pt, lang: "en")
#set par(leading: 0.5em, spacing: 0.6em)
#set list(indent: 0.6em, body-indent: 0.5em, spacing: 0.45em)
#set enum(spacing: 0.75em)
#show link: set text(fill: black)

#let header-text(body) = text(font: header-font, body)
#let title-text(body) = text(font: title-font, body)

// Bold, slightly larger emphasis (LaTeX \emph override)
#let em(body) = text(size: 11pt, weight: "bold", body)

// Two-column section: title on the left, content on the right
#let section(title, body) = {
  grid(
    columns: (title-width, 1fr),
    column-gutter: 0pt,
    title-text(text(size: 12pt, smallcaps(par(leading: 0.35em, title)))),
    box(width: content-width, body),
  )
  v(25pt)
}

// Two-row heading: institution / place, degree / dates
#let edu-heading(inst, place, degree, dates) = {
  block(spacing: 0.45em, below: 0.45em)[
    #em(inst) #h(1fr) #place \
    #emph(degree) #h(1fr) #dates
  ]
}

// One-row heading: name | role, dates
#let intern-heading(name, role, dates) = {
  block(spacing: 0.45em, below: 0.45em)[#em(name) | #role #h(1fr) #dates]
}

// Project heading: bold title, then funder / dates
#let proj-heading(name, funder, dates) = {
  block(spacing: 0.7em, below: 0.6em)[
    *#name* \
    #emph(funder) #h(1fr) #dates
  ]
}

#let item-list(..items) = block(below: 0.9em, list(..items))
#let pub-list(..items) = {
  set par(justify: true)
  enum(indent: 0pt, ..items)
}
#let plain-list(..items) = {
  set par(justify: true)
  stack(spacing: 0.6em, ..items.pos())
}
#let latex = box[L#h(-0.36em)#text(size: 0.7em, baseline: -0.22em)[A]#h(-0.15em)T#h(-0.1667em)#box(move(dy: 0.22em)[E])#h(-0.125em)X]
#let subhead(body) = block(above: 0.9em, below: 0.6em, strong(body))

// Icons: GitHub and Google Scholar from Simple Icons (CC0), LinkedIn from Bootstrap Icons (MIT)
#let icon-github = ``` <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12"/></svg>```.text
#let icon-scholar = ``` <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M5.242 13.769L0 9.5 12 0l12 9.5-5.242 4.269C17.548 11.249 14.978 9.5 12 9.5c-2.977 0-5.548 1.748-6.758 4.269zM12 10a7 7 0 1 0 0 14 7 7 0 0 0 0-14z"/></svg>```.text
#let icon-linkedin = ``` <svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg"><path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854zm4.943 12.248V6.169H2.542v7.225zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248S2.4 3.226 2.4 3.934c0 .694.521 1.248 1.327 1.248zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016l.016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225z"/></svg>```.text

#let icon-link(svg, url, label) = box(link(url)[#box(baseline: 0.12em, image(bytes(svg), format: "svg", height: 0.85em)) #label])

// -------------------- HEADING --------------------
#grid(
  columns: (1fr, auto),
  column-gutter: 1em,
  align: (left + bottom, right + bottom),
  header-text[
    #text(size: 30pt, smallcaps[Pablo Miralles])
    #h(1em)
    #text(size: 14.5pt, smallcaps[Ph.D. Candidate])
  ],
  header-text(stack(
    spacing: 0.55em,
    link("mailto:pablo.miralles@upm.es")[pablo.miralles\@upm.es],
    link("https://pablomirallesg.com")[https://pablomirallesg.com],
    text(size: 10pt)[
      #icon-link(icon-scholar, "https://scholar.google.com/citations?user=l538kF8AAAAJ")[Scholar]
      #h(0.4em)
      #icon-link(icon-github, "https://github.com/pablomiralles22")[GitHub]
      #h(0.4em)
      #icon-link(icon-linkedin, "https://www.linkedin.com/in/pablo-miralles-gonzalez/")[LinkedIn]
    ],
  )),
)
#v(0.1em)
#line(length: 100%, stroke: 0.4pt)
#v(1.2em)

// -------------------- RESEARCH INTERESTS --------------------
#section[Research \ Interests][
  #set par(justify: true)
  #h(1.5em)I work on adapting large language models to downstream tasks in low-resource settings in a way that generalizes. My work combines parameter-efficient reinforcement-learning fine-tuning (QLoRA + GRPO for natural language inference) with training-free methods built on LLM log-probabilities (AI-generated text detection, authorship attribution and verification), with a focus on robustness and out-of-distribution generalization.
]

// -------------------- EDUCATION --------------------
#section[Education][
  #edu-heading[Technical University of Madrid][Madrid, Spain][Ph.D. in Artificial Intelligence][2024 -- Now]
  #item-list(
    [Advisor: Prof. Alejandro Martín. Expected defence: September 2027.],
    [Thesis theme: adapting LLMs to low-resource downstream tasks in a way that generalizes (working title pending).],
  )

  #edu-heading[Technical University of Madrid][Madrid, Spain][M.Sc. in Machine Learning and Big Data][2023 -- 2024]
  #item-list[GPA: 9.65/10.00, Rank: 1st]

  #edu-heading[University of Murcia][Murcia, Spain][B.Sc. in Mathematics][2017 -- 2022]
  #item-list[GPA: 9.74/10.00, Rank: 1st]

  #edu-heading[University of Murcia][Murcia, Spain][B.Sc. in Computer Science][2017 -- 2022]
  #item-list[GPA: 9.57/10.00, Rank: 2nd]
]

// -------------------- RESEARCH VISITS --------------------
#section[Research \ Visits][
  #edu-heading[Queen Mary University of London][London, UK][Visiting Ph.D. Researcher][2026.09 -- 2026.12]
  #item-list[Host: Prof. Arkaitz Zubiaga. Research on multimodal hate speech detection.]
]

// -------------------- PUBLICATIONS --------------------
#let me = [*Miralles-González, P.*]
#let code(url) = [#h(0.3em)#icon-link(icon-github, url)[GitHub]]

#section[Publications][
  #subhead[Conference papers]
  #pub-list(
    [#me, Huertas-Tato, J., Martín, A., & Camacho, D. (2026). One-Shot Style Transfer LLM Log-Probabilities for Authorship Attribution and Verification. _Findings of the Association for Computational Linguistics: EMNLP 2026_, 278--297, Budapest, Hungary. #link("https://doi.org/10.18653/2026.findings-emnlp.14") #code("https://github.com/pablomiralles22/OSST")],
  )

  #subhead[Journal articles]
  #pub-list(
    [#me, Huertas-Tato, J., Martín, A., & Camacho, D. (2026). Evaluating the scalability and adversarial generalization of GRPO-trained NLI models. _Knowledge-Based Systems_, 352, 117013. #link("https://doi.org/10.1016/j.knosys.2026.117013") #code("https://github.com/pablomiralles22/paper-nli-grpo")],
    [#me, Huertas-Tato, J., Martín, A., & Camacho, D. (2026). Not all tokens are created equal: Perplexity Attention Weighted Networks for AI-generated text detection. _Information Fusion_, 125, 103465. #link("https://doi.org/10.1016/j.inffus.2025.103465") #code("https://github.com/pablomiralles22/ai-gen-detection")],
    [#me, Huertas-Tato, J., Martín, A., & Camacho, D. (2026). On the locality bias and results in the Long Range Arena. _IEEE Transactions on Emerging Topics in Computational Intelligence_ (accepted, to appear). #link("https://doi.org/10.48550/arXiv.2501.14850") #code("https://github.com/pablomiralles22/paper-LRA-source")],
  )

  #subhead[Preprints]
  #pub-list(
    [Girón, A., #me, Huertas-Tato, J., D'Antonio, S., & Camacho, D. (2026). xList-Hate: A Checklist-Based Framework for Interpretable and Generalizable Hate Speech Detection. _arXiv_ (under review). #link("https://doi.org/10.48550/arXiv.2602.05874")],
  )
]

// -------------------- TEACHING --------------------
#section[Teaching][
  #intern-heading[Generative Models][M.Sc. in Deep Learning (Instructor)][2024/25 -- present]
  #item-list[VAEs, GANs, autoregressive and diffusion models. Co-designed syllabus and materials (50%) with another instructor; 15--30 students per edition.]

  #intern-heading[Deep Learning for NLP][M.Sc. in Deep Learning (Instructor)][2024/25 -- present]
  #item-list[Real-world NLP applications with HuggingFace, LangGraph, vLLM and vector databases. Co-designed syllabus and materials (50%) with another instructor; 15--30 students per edition.]

  #intern-heading[Data Structures][B.Sc. in Computer Science (Teaching assistant)][2025/26 -- present]
  #item-list[Queues, stacks, lists, trees, graphs and hash tables.]

  #intern-heading[Classical Prediction Methods][B.Sc. in Data Science (Teaching assistant)][2025/26]
  #item-list[Lectures and practical sessions on linear regression, clustering and time series models.]

  #intern-heading[Algorithms and Data Structures][B.Sc. in Data Science (Teaching assistant)][2024/25]
  #item-list[Graph theory fundamentals and greedy algorithm design.]
]

// -------------------- MENTORING AND SERVICE --------------------
#section[Mentoring \ and \ Service][
  #plain-list(
    [#em[Supervision]: Co-supervised 5--6 M.Sc. theses.],
    [#em[Seminars]: Co-organizer of the weekly research group seminar series (2025/26): members' research talks and tutorials on tools and topics such as LLM efficiency.],
    [#em[Peer review]: IDEAL (2 papers), The Computer Journal (1 paper).],
  )
]

// -------------------- PROJECTS --------------------
#section[Projects and grants][
  #proj-heading[FPU Fellowship (University Teacher Training Programme)][Spanish Ministry of Science and Innovation][2025.01 -- Now]
  #item-list[Highly competitive predoctoral contract aimed at training future university lecturers and researchers.]

  #proj-heading[XAI-DisInfodemics: eXplainable AI for disinformation and conspiracy detection during infodemics][European Union NextGenerationEU/PRTR][2024.05 -- 2025.01]

  #proj-heading[FightDIS: Fighting against Information DISorders in Online Social Networks][Spanish Ministry of Science and Innovation][2023.09 -- 2024.05]
]

// -------------------- AWARDS & HONORS --------------------
#section[Awards \ and \ Honors][
  #set list(indent: 0pt, spacing: 0.6em)
  #set par(justify: true)
  - #em[Best academic record], M.Sc. in Machine Learning and Big Data #h(1fr) 2024
  - #em[Extraordinary End of Degree Award], B.Sc. in Mathematics and B.Sc. in Computer Science #h(1fr) 2023
  - #em[Academic Excellence Award in joint B.Sc. programs], Math + CS #h(1fr) 2023
  - #em[SWERC (ICPC regional)]: 18th, 17th and 23rd place, 2019/20--2021/22 #h(1fr) 2020--2022
]

// -------------------- SKILLS --------------------
#section[Skills][
  #plain-list(
    [#em[Languages]: Spanish (native), English (C1).],
    [#em[Programming]: Python, C++, Java.],
    [#em[Deep Learning]: PyTorch, HuggingFace, vLLM, QLoRA, GRPO, Triton (basic), CUDA (basic).],
    [#em[Data Science/Machine Learning]: scikit-learn, NumPy, Pandas.],
    [#em[Experiment management/distributed]: Weights & Biases, Ray.],
    [#em[Other]: Docker, Linux, Git, #latex.],
  )
]

// -------------------- INDUSTRY --------------------
#section[Industry Experience][
  #intern-heading[Fever][Software Engineer][2023.02 -- 2023.08]
  #item-list[Led a project using LLMs to automatically structure information and generate plan content, substantially reducing content production costs.]
]

// // -------------------- REFERENCES --------------------
// #section[References][
//   #plain-list(
//     [Prof. Alejandro Martín (Ph.D. advisor), Technical University of Madrid.],
//     [Prof. David Camacho, Technical University of Madrid.],
//   )
// ]
