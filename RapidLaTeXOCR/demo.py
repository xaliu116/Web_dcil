
from rapid_latex_ocr import LatexOCR

import sys
import logging
import traceback

from docx.oxml.ns import qn
from docx import Document
from lxml import etree
import Converter as l2m

from config import ROOT_PATH

mml2omml = ROOT_PATH + './RapidLaTeXOCR/MML2OMML2.xsl'
#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(filename='latex.log',level=logging.DEBUG)
model = LatexOCR()


def ppmathml(mathml, type='string'):
    if type == 'string':
        mathml = etree.fromstring(mathml)
    elif type != 'xml':
        raise TypeError('Expected types string or xml')
    pp = etree.tostring(mathml, pretty_print=True, encoding='unicode')
    print(pp)
    return pp


class LatexConverter:
    def __init__(self):
        xslt = etree.parse(mml2omml)
        self.transform = etree.XSLT(xslt)

    def convert(self, latex_input):
        logging.debug(latex_input)
        latex_input = self.handle_custom_commands(latex_input)
        logging.debug(latex_input)
        mathml_string = l2m.latex_to_mathml(latex_input, name_space=True)
        print(mathml_string)
        # mathml_string = etree.tostring(mathml_tree)
        logging.debug(mathml_string)
        mathml_tree = etree.fromstring(mathml_string)
        omml_tree = self.transform(mathml_tree)
        print(self.transform.error_log)
        return omml_tree.getroot()

    @staticmethod
    def handle_custom_commands(latex):
        nagwa_commands = {
            'nagwaMatrix': 'pmatrix',
            'nagwaCases': 'cases'
        }
        for nagwa_command, substitute in nagwa_commands.items():
            latex = latex.replace(nagwa_command, substitute)
        return latex






if __name__ == '__main__':

    img_path = "tests/test_files/5.png"   #输入
    with open(img_path, "rb") as f:
        data = f.read()
    latexres, elapse = model(data)
    '''
    以上为公式识别
    '''

    converter = LatexConverter()
    summary = []
    print('Test for latex input: - "{}" \n'.format(latexres))
    try:
        eqn = converter.convert(latexres)
        x = etree.tostring(eqn, pretty_print=True, encoding='unicode')
        doc = Document()
        paragraph = doc.add_paragraph(style=None)
        paragraph._element.append(eqn)
        docx_path = '{}.docx'.format('test')
        doc.save(docx_path)
        summary.append('- Success for latex input -  "{}"'.format(latexres))
    except Exception as e:
        exc_info = sys.exc_info()
        print(e)
        traceback.print_exception(*exc_info, file=sys.stdout)
        del exc_info
        summary.append('- Falure for latex input - "{}"'.format(latexres))
    print('\n'.join(summary))