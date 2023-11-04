# For preparing data for minlizhash

# import trafilatura as tr
# from tokens import Document
# from trafilatura.external import try_justext, try_readability
# from warcio.archiveiterator import ArchiveIterator

# @dataclass
# class WebDocument(Document):
#     url: str
#     html: str
#     text: str
#     @classmethod
#     def parse_record(cls, record: ArchiveIterator) -> WebDocument:
#         return WebDocument(
#             url=record.rec_headers.get_header("WARC-Target-URI"),
#             html=record.content_stream().read().decode("utf-8"),
#             text=tr.extract(
#                 record.content_stream().read().decode("utf-8"),
#                 favor_precision=True,
#                 include_comments=False,
#                 #  output_format="txt",
#                 include_formatting=False,
#                 tei_validation=True,
#                 output_format="txt",
#             ),
#         )
