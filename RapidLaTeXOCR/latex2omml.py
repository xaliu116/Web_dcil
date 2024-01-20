
import latex2mathml.converter

import sys
import re
import logging
import traceback
import pprint

from docx import shared
from docx.oxml.ns import qn
from docx import Document
from lxml import etree
import Converter as l2m

mml2omml = 'MML2OMML2.xsl'
#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(filename='latex.log',level=logging.DEBUG)
        
    
def ppmathml(mathml, type = 'string'):
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
        mathml_string = l2m.latex_to_mathml(latex_input, name_space = True)
        print(mathml_string)
        #mathml_string = etree.tostring(mathml_tree)
        logging.debug(mathml_string)
        mathml_tree = etree.fromstring(mathml_string)
        omml_tree = self.transform(mathml_tree)
        print(self.transform.error_log)
        return omml_tree.getroot()
        
 
        
    @staticmethod
    def handle_custom_commands(latex):
        nagwa_commands = {
            'nagwaMatrix': 'pmatrix',
            'nagwaCases' : 'cases'
            } 
        for nagwa_command, substitute in nagwa_commands.items():
            latex = latex.replace(nagwa_command, substitute)
        return latex


def _test_mathml():
    test_batch = [r'f\left(x\right) \
  =a_{0}+\sum_{n=1}^{\infty}\left(a_{n}\cos(\frac{n\pi x}{L}) \
  +b_{n}\sin(\frac{n\pi x}{L})\right)']
    converter = LatexConverter()
    summary = []
    for i, latex_input in enumerate(test_batch):
        print(('*' * 50) + '  Start of Test {} '.format(i+1) + ('*' * 50))
        print('Test for latex input: - "{}" \n'.format(latex_input))
        
        #print(converter._strip_align(latex_input))
        #x = etree.tostring(converter.convert(latex_input), pretty_print=True, encoding='unicode')
        #print(x)
        
        try:
            eqn = converter.convert(latex_input)
            x = etree.tostring(eqn, pretty_print=True, encoding='unicode')
            doc = Document()

            paragraph = doc.add_paragraph(style=None)
            paragraph._element.append(eqn)
            docx_path = '{}.docx'.format('test')
            doc.save(docx_path)
            print(x)
            print(('+' * 50) + '  Test Success  ' + ('+' * 50))
            summary.append('{} - Success for latex input -  "{}"'.format(i+1, latex_input))

        except Exception as e:
            exc_info = sys.exc_info()
            print(('-' * 50) + '  Test Failure  ' + ('-' * 50))
            print(e)
            traceback.print_exception(*exc_info, file=sys.stdout)
            del exc_info
            summary.append('{} - Falure for latex input - "{}"'.format(i+1, latex_input))

    print('\n'.join(summary))


    
if __name__ == '__main__':
    _test_mathml()