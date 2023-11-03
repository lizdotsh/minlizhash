import pickle

import numpy as np
import tiktoken
import trafilatura as tr
from trafilatura.external import try_justext, try_readability
from warcio.archiveiterator import ArchiveIterator

enc = tiktoken.get_encoding("cl100k_base")
records = []
text = []
with open(
    "commoncrawl/2023-09/CC-MAIN-20230921073711-20230921103711-00000.warc", "rb"
) as stream:
    n = 0
    for record in ArchiveIterator(stream):
        if record.rec_type == "response":
            try:
                extract = tr.extract(
                    record.content_stream().read().decode("utf-8"),
                    favor_precision=True,
                    include_comments=False,
                    #  output_format="txt",
                    include_formatting=False,
                    tei_validation=True,
                    output_format="txt",
                )
                records.append(np.array(enc.encode_ordinary(extract)))
            except:
                print("error")
                continue
            n += 1
            # Do something with the record
            # text.append(html2txt(record.content_stream().read()))

        if n == 2000:
            break
# records[38]
#
# records[9].decode("utf-8")
try_justext, try_readability


def get_text(num):
    return tr.extract(
        records[num],
        favor_precision=True,
        include_comments=False,
        #  output_format="txt",
        include_formatting=False,
        tei_validation=True,
        output_format="txt",
    )


with open("my_list.pkl", "wb") as f:
    pickle.dump(records, f)
