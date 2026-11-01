# Recommended learning books and source links

A machine-readable version of this list is maintained in `knowledge_base/metadata/books_manifest.json`.

The LearnML application starts with original starter notes. You can add more source material in `knowledge_base/documents/` and rebuild the index from the Knowledge Library page.

## Recommended sources

| Resource | Topics | Official link | Licence / reuse note |
|---|---|---|---|
| Dive into Deep Learning | ML foundations, neural networks, CNNs, RNNs, attention, Transformers, NLP, reinforcement learning, generative models | https://d2l.ai/ · PDF: https://d2l.ai/d2l-en.pdf | CC BY-SA 4.0. The helper script `download_books.py` downloads the PDF from the official site. Keep attribution and the ShareAlike notice. |
| OpenIntro Statistics, 4th edition | Data, distributions, probability, inference, regression | https://www.openintro.org/book/os/ | CC BY-SA 3.0 for most OpenIntro Statistics resources. The official book page links to the free PDF; choose a $0 contribution on the download page. |
| Introduction to Modern Statistics, 2nd edition | Data science, EDA, regression, statistical inference | https://openintrostat.github.io/ims/ | CC BY-SA 3.0 Unported, as stated on the book website. Free web edition and PDF option. |
| Mathematics for Machine Learning | Linear algebra, analytic geometry, matrix decompositions, calculus, probability, optimisation, PCA, regression, SVM | https://mml-book.github.io/ | The authors make a PDF freely available. I could not verify an explicit open redistribution licence from the official book page, so this project links to it but does not redistribute it. |
| The Turing Way | Reproducible, ethical and collaborative data science | https://book.the-turing-way.org/ | CC BY 4.0. It is a handbook on research/data-science practice, not a core ML algorithms textbook. |
| Python Data Science Handbook | Python, NumPy, Pandas, Matplotlib, scikit-learn | https://jakevdp.github.io/PythonDataScienceHandbook/ | The book text is CC BY-NC-ND. Link to the official online edition; do not repackage or modify the book text without permission. |

## Attribution notes

- OpenIntro states that most of its Statistics resources are CC BY-SA 3.0. Some resources have exceptions, so check the licence for the specific file before redistributing it.
- *Dive into Deep Learning* is CC BY-SA 4.0. Any redistributed adapted material must observe the licence's attribution and ShareAlike conditions.
- Do not add copyrighted commercial textbooks to this repository just because they can be found online.

## Add a book

1. Download the book from its official source.
2. Confirm that its licence permits your intended use and redistribution.
3. Place the PDF in `knowledge_base/documents/`.
4. Open LearnML → Knowledge Library → Build knowledge index.
5. The index stores extracted text and page metadata; the original PDF remains local and is excluded from Git by default.

For GitHub and Streamlit deployment, build the index locally and commit `storage/chunks.json` and `storage/embeddings.npy`. That allows the deployed app to use the index without storing large book PDFs in the Git repository. Keep the original PDF files locally or download them again with the official-source helper.
