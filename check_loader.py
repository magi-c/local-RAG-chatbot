from rag.loader import load_pdf
pages = load_pdf("data/EU_Directive_Accessibility.pdf")
print(len(pages), pages[0].text[:300])