import zipfile
import xml.etree.ElementTree as ET

def read_docx(path):
    with zipfile.ZipFile(path) as z:
        xml_content = z.read("word/document.xml")
    tree = ET.fromstring(xml_content)
    text = []
    for elem in tree.iter():
        if elem.tag.endswith("t"):
            if elem.text:
                text.append(elem.text)
        elif elem.tag.endswith("p"):
            text.append("\n")
    return "".join(text)

print("=== READING review 1.docx ===")
review_text = read_docx(r"C:\Users\Shourya Thakur\Downloads\review 1.docx")
with open("review_1_extracted.txt", "w", encoding="utf-8") as f:
    f.write(review_text)
print("review 1 length:", len(review_text))

print("=== READING team-14-deepfake-detection-itr (1).docx ===")
itr_text = read_docx(r"C:\Users\Shourya Thakur\Downloads\team-14-deepfake-detection-itr (1).docx")
with open("itr_extracted.txt", "w", encoding="utf-8") as f:
    f.write(itr_text)
print("itr length:", len(itr_text))
