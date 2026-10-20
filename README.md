# AChemS Research Atlas

Interactive Streamlit research prototype for AChemS conference programs and abstracts (1979–2026).

## Deploy
In [Streamlit Community Cloud](https://share.streamlit.io), choose Create app, select `Wan2bg/achems-research-atlas`, branch `main`, main file `app.py`, then deploy.

## Import real data
Download PDFs from the [AChemS Drive archive](https://drive.google.com/drive/folders/1EVbIZI7RNa5dd4HncUhwaBS062l7-l6i) and upload them in the Import PDFs tab. Text extraction finds unverified candidate lines, not confirmed abstracts or titles. Scanned PDFs require OCR. Upload a reviewed CSV with `year,title,authors,abstract,institution,presentation_type,source_file,source_page,verified` for more reliable analysis.

## Research limitations
The app has no fabricated presentation counts. It does not automatically access private Drive content. Charts and machine-learning clusters use uploaded records only and need validation. Download edited CSV files because session storage is temporary. Respect archive rights before publicly distributing full PDFs.
