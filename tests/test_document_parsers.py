from email.message import EmailMessage

import openpyxl

from build_index import load_document_file


def test_load_excel_document(tmp_path):
    path = tmp_path / "case.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Cases"
    sheet.append(["id", "summary"])
    sheet.append(["A-1", "payment dispute"])
    workbook.save(path)

    documents = load_document_file(path)
    assert "payment dispute" in documents[0].page_content
    assert documents[0].metadata["sheet"] == "Cases"


def test_load_email_document(tmp_path):
    path = tmp_path / "notice.eml"
    message = EmailMessage()
    message["Subject"] = "Case update"
    message["From"] = "sender@example.com"
    message.set_content("The case has been updated.")
    path.write_bytes(message.as_bytes())

    documents = load_document_file(path)
    assert "Case update" in documents[0].page_content
    assert "updated" in documents[0].page_content
