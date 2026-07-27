import unittest

from rag_app.infrastructure.parsing.document_parser import DocumentParser


class DocumentParserSpecialExtensionTest(unittest.TestCase):
    def setUp(self):
        self.parser = DocumentParser()

    def test_att_text_file_is_supported_and_parsed(self):
        chunks = self.parser.parse("长63渗透率.att", "渗透率 12.5 mD".encode("utf-8"))

        self.assertTrue(self.parser.supports("长63渗透率.att"))
        self.assertEqual(len(chunks), 1)
        self.assertIn("渗透率 12.5 mD", chunks[0].text)

    def test_gdb_binary_file_is_supported_and_keeps_file_name(self):
        chunks = self.parser.parse("长63渗透率.gdb", b"\x00\x01permeability=12.5\x00\x02")

        self.assertTrue(self.parser.supports("长63渗透率.GDB"))
        self.assertEqual(len(chunks), 1)
        self.assertIn("长63渗透率.gdb", chunks[0].text)
        self.assertIn("permeability=12.5", chunks[0].text)


if __name__ == "__main__":
    unittest.main()
