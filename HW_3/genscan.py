import requests
import re
from bs4 import BeautifulSoup
from dataclasses import dataclass, field


@dataclass
class GenscanOutput:
    status: int
    cds_list: list = field(default_factory=lambda: []),
    intron_list: list = field(default_factory=lambda: []),
    exon_list: list = field(default_factory=lambda: [])


@dataclass
class Splice:
    number: str
    start: int
    end: int


class BadRequest(Exception):

    def __init__(self, response):
        self.response = response

    def __repr__(self):
        return f"Request {self.response.url} failed with status {self.response.status_code}"


def run_genscan(sequence=None,
                sequence_file=None,
                organism="Vertebrate",
                exon_cutoff=1.00,
                sequence_name=""):
    if sequence_file:
        file = open(sequence_file, 'r')
        files = {'-u': file}
    else:
        file = None
        files = {}

    data = {
        '-o': organism,
        '-e': exon_cutoff,
        '-n': sequence_name,
        '-s': sequence,
        '-p': 'Predicted peptides only'
    }
    response = requests.post('http://hollywood.mit.edu/cgi-bin/genscanw_py.cgi', data=data, files=files)
    if response.status_code in range(400, 500):
        raise BadRequest(response)
    if file:
        file.close()

    content = BeautifulSoup(response.content, "lxml").find('pre').text

    cds_unformatted = re.findall(pattern=r'^>.*\n\n([A-Z\n]+)', string=content, flags=re.MULTILINE)

    cds = list(map(lambda protein: protein.strip().replace('\n', ''), cds_unformatted))

    exons_section = re.search(pattern=r'Predicted genes/exons:([\n\w\./ \-+]+)Suboptimal exons with probability', string=content).group(1)
    exons = re.findall(pattern=r'^(?!PlyA)\d[\.\w\+\- ]+', string=exons_section, flags=re.MULTILINE)
    exons = filter(lambda exon: 'PlyA' not in exon, exons)

    def splice_from_exon(exon):
        data = re.split(r' +', exon)
        return Splice(data[0], int(data[3]), int(data[4]))

    exons = list(map(splice_from_exon, exons))

    introns = []
    for i in range(len(exons) - 1):
        curr_exon = exons[i]
        next_exon = exons[i+1]
        introns.append(Splice(curr_exon.number, curr_exon.end + 1, next_exon.start - 1))

    return GenscanOutput(response.status_code, cds, exons, introns)
