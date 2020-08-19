# -*- coding: utf-8 -*-
""" Test dteulaapp library. """

# Standard library imports
import unittest
import os
import base64

# dteulaapp imports
import dteulaapp.core as DTEULAAPP
import dteulaapp.helpers as helpers
from dteulaapp import create_app

clausetext1 = """
MASTER SERVICES SUBSCRIPTION AGREEMENT



Parties:	
“COMPANY”	
“Client” or “Ordering Activity”
Full Legal Name:	COMPANY INC.	<< INSERT FULL CLIENT NAME >>
Business Entity Type:	Corporation	
Organized In:	STATE	
Address:	ADDRESS	<< Insert address >>
<< Insert address >>
	Attn:    		Attn:  	
	Phone:   		Phone:	 	
	Email address:  		Email address:  	
Agreement Effective Date:    	


This Master Services Subscription Agreement (the “Agreement”) sets forth the terms and conditions governing COMPANY’ provision to Client of a cloud-based asset management and decision support system.
This Agreement, including the Order Form attached to it, as well as any Order Forms and Statements of Work entered into by the parties from time to time, the underlying AGENCY Schedule Contract, and Schedule Pricelist, together constitute the entire agreement of the parties and supersede any prior and contemporaneous oral or written understanding as to the parties’ relationship and the subject matter hereof. In the event of any conflict or contradiction among the foregoing documents, the documents will control in the order listed in Contract Clause 552.212-4(s). This Agreement may only be amended in a writing signed by both parties.
This Agreement may be executed in two or more counterparts, each of which will be deemed an original for all purposes, and together will constitute one and the same document. Once signed, both parties agree that any reproduction of this Agreement made by reliable means (for example, a photocopy, facsimile, or PDF file) is an original.


AGREED TO AND ACCEPTED:
“COMPANY"		AGREED TO AND ACCEPTED:
“Client”	
COMPANY INC.		<< INSERT FULL CLIENT NAME >>	
Authorized Signature		Authorized Signature	
Print Name		Print Name	
Title		Title	
Date	Date
 

1.	Definitions.
a.	“Client Data” refers to the data Client or any User uploads or otherwise supplies to, or stores in, the Services under Client’s account.
b.	“Documentation” means the user guides, help information, and other technical and operations manuals and specifications for the Services made available by COMPANY in electronic or other form, as updated from time to time.
c.	“Order Form” refers to the document which specifies, among other things, the Services to be provided to Client by COMPANY as well as the scope of use, order effective date and term, Subscription Fees and other prices, billing period, and other applicable details. The initial Order Form entered into by the parties is attached to this Agreement. To be effective, additional Order Forms must be signed by both parties. All Order Forms are subject to this Agreement.
d.	“Services” refers to the online integrated asset management and decision support system offerings, and any other product or service provided to Client by COMPANY as specified on the applicable Order Form.
e.	“Subscription Fees” mean the fees paid by Client in accordance with the AGENCY Pricelist for the right to access and use the Services during the applicable Service Term.
f.	“User" refers to each employee, agent, contractor, and consultant who is authorized by Client to use the Services, and has been supplied a user identification and password by Client (or by COMPANY at Client’s request).
2.	Provision of the Services.
a.	Availability and Use of the Services. COMPANY will make the Services available to Client in accordance with each Order Form entered into by the parties. Client’s use of the Services is limited to its internal business purposes solely for the scope and use limitations specified in the applicable Order Form.
b.	Changes to the Services. COMPANY may make changes, modifications and enhancements to the Services from time to time. Any material change, modification, or enhancement of the service agreed to in the Order Form and this Agreement shall be presented to Client for review and will not be effective unless and until both parties sign a written agreement updating these terms.
c.	Support for the Services. COMPANY will provide Client with the support described in COMPANY’ then current technical support policy, a current copy of which is attached as Exhibit A.
d.	Business Continuity and Security Measures for the Services.
i.	Data Backup. The Services include standard off-site backup and recovery capabilities including daily incremental backups with synthetic full backups created weekly and monthly. Weekly and monthly full backups are stored off-site on disk or via a cloud data storage service. With respect to long term retention, COMPANY follows industry standard best practices in having 24 monthly and seven yearly backups. Upon request, COMPANY will offer additional long term monthly and yearly data retention options tailored to address unique Client requirements.
ii.	Data Restoration. In the event of a loss of Client Data due to a disaster, the Data is restored using the most recent backup so that the Services are available within twelve hours of the incident. In the event of a server (host) loss, an already “imaged” stand-by server will be provisioned in place of the failed server in the “state-less” application server farm. This standby server can be in production within four hours.
iii.	Business Continuity. COMPANY’ business continuity plans adhere to industry best practices. COMPANY will invoke those plans in the event there is a clearly adverse impact to the Services. COMPANY will review its business continuity plan for disaster recovery on an annual basis at Client’s request including any changes that have been made to the plan since the prior review. COMPANY will also ensure that any changes to its business continuity plans are communicated to Client in the event of any material change to the plan.
iv.	Security Measures. In providing the Services, COMPANY complies with its information security procedures. A current copy of these procedures is attached as Exhibit B. COMPANY will provide Client on an annual basis with SSAE16 Reviews from the third-party data center providers utilized in the provision of the Services to Client. Client acknowledges and agrees that all SSAE16 Reviews constitute Confidential Information of COMPANY. COMPANY recognizes that Federal agencies are subject to the Freedom of Information Act, 5
U.S.C. 552, which requires that certain information be released, despite being characterized as “confidential” by the vendor.
3.	Order Process. Client will order Services by signing an Order Form. In the event that Client’s business practices require a purchase order number be issued prior to payment of any COMPANY invoices issued pursuant to an Order Form, Client will promptly provide that number to COMPANY. Except as set forth in the Order Form, terms, provisions or conditions on any purchase order, acknowledgement, or other business form or writing that Client may provide to COMPANY or use in connection with the procurement of Services (or any software) from COMPANY will have no effect on the rights, duties or obligations of the parties under this Agreement, regardless of any failure of COMPANY to object to such terms, provisions or conditions.
4.	Professional Services. If professional services (such as implementation, training, consulting, etc.) are included in any Order Form (“Professional Services”), then they will be set forth in a separately executed Statement of Work (“Statement of Work”) containing relevant project details including, if applicable, any works to be developed by COMPANY and provided to Client (“Deliverables”). In addition, the following provisions will apply to all Statements of Work: (a) COMPANY will retain all ownership rights to any and all Deliverables excluding, any pre-existing technology, materials or Client Confidential Information supplied by Client for incorporation into such Deliverable; and (b) COMPANY grants Client a royalty-free, non-exclusive, non-transferable, non-assignable worldwide license to use any

"""

class ClauseExtractTest(unittest.TestCase):
  """ Can clauses be extracted well from EULA text? """
  def test(self):
    theclauses = DTEULAAPP.extractClauses(clausetext1)
    self.assertEqual(len(theclauses), 22, 'There are 22 clauses in the text')

class CleanTextTest(unittest.TestCase):
  """ Can a text be cleaned up? """
  def test(self):
    text1 = 'a.\t“Client Data” refers to the data Client or any User uploads or otherwise supplies to, or stores in, the Services under Client’s account.'
    good1 = 'a. "Client Data" refers to the data Client or any User uploads or otherwise supplies to, or stores in, the Services under Client\'s account.'
    result1 = helpers.cleantext(text1)
    self.assertEqual(result1, good1, 'The text should not have tabs or smart quotes')

class ClauseStartText(unittest.TestCase):
  """ Can a clause starter be detected in a text? """
  def test(self):
    text1 = 'a. "Client Data" refers to the data Client or any User uploads or otherwise supplies to, or stores in, the Services under Client\'s account.'
    text2 = 'ii. "Client Data" refers to the data Client or any User uploads or otherwise supplies to, or stores in, the Services under Client\'s account.'
    text3 = '8. "Client Data" refers to the data Client or any User uploads or otherwise supplies to, or stores in, the Services under Client\'s account.'
    text4 = '"Client Data" refers to the data Client or any User uploads or otherwise supplies to, or stores in, the Services under Client\'s account.'
    self.assertTrue(helpers.clausestarter(text1), 'Text 1 starts a clause')
    self.assertTrue(helpers.clausestarter(text2), 'Text 2 starts a clause')
    self.assertTrue(helpers.clausestarter(text3), 'Text 3 starts a clause')
    self.assertFalse(helpers.clausestarter(text4), 'Text 4 does not start a clause')

class PDFParseTest(unittest.TestCase):
  """ Can a PDF file be parsed correctly? """
  def test(self):
    resourcefolder = helpers.getresourcefolder()
    pdfPath = os.path.join(resourcefolder, 'sample_eula_1.pdf')
    curdata = ''
    with open(pdfPath, 'rb') as curfile:
      curdata = base64.b64encode(curfile.read()).decode()
    plaintext = DTEULAAPP.parsepdf(curdata)
    self.assertEqual(len(plaintext), 45812, 'The sample PDF EULA should have 45812 characters of plain text')

class PDFClauseExtractTest(unittest.TestCase):
  """ Can clauses be extracted from a PDF file correctly? """
  def test(self):
    resourcefolder = helpers.getresourcefolder()
    pdfPath = os.path.join(resourcefolder, 'sample_eula_1.pdf')
    curdata = ''
    with open(pdfPath, 'rb') as curfile:
      curdata = base64.b64encode(curfile.read()).decode()
    plaintext = DTEULAAPP.parsepdf(curdata)
    clauses = DTEULAAPP.extractPDFClauses(plaintext)
    self.assertEqual(len(clauses), 134, 'The sample PDF EULA should have 134 clauses')

class WordParseTest(unittest.TestCase):
  """ Can a Word file be parsed correctly? """

  def test(self):
    resourcefolder = helpers.getresourcefolder()
    wordPath = os.path.join(resourcefolder, 'sample_eula_1.docx')
    curdata = ''
    with open(wordPath, 'rb') as curfile:
      curdata = base64.b64encode(curfile.read()).decode()
    plaintext = DTEULAAPP.parseword(curdata)
    self.assertEqual(len(plaintext), 47608, 'The sample Word EULA should have 47608 characters of plain text')

class AppCreationTest(unittest.TestCase):
  """ Can the app be initialized properly? """

  def test(self):
    testapp = create_app()
    self.assertIsNotNone(testapp, 'The app should be created successfully')
    self.assertEqual(len(testapp.blueprints), 3, 'The app should have 3 blueprints')
    self.assertIsNotNone(DTEULAAPP.eulamodel, 'The app should have loaded the model')

class PredictTest(unittest.TestCase):
  """ Can clauses be extracted well from EULA text? """
  def test(self):
    DTEULAAPP.loadmodels()
    theclauses = DTEULAAPP.extractClauses(clausetext1)
    predictions = DTEULAAPP.predicteula(theclauses)
    self.assertEqual(len(predictions['prediction']), 22, 'There are 22 predictions')
    self.assertEqual(len(predictions['windows']), 22, 'There are 22 sets of windows')

class DiceTest(unittest.TestCase):
  """ Can clauses be searched using Dice? """
  def test(self):
    DTEULAAPP.loadmodels()
    theclauses = DTEULAAPP.extractClauses(clausetext1)
    matches = [DTEULAAPP.dicesearch(curclause, DTEULAAPP.accsearch) for curclause in theclauses]
    self.assertEqual(len(matches), 22, 'There are 22 matches')
    self.assertEqual(len(matches[0]), 83, 'The first clause matched a known acceptable clause of 83 characters in length')

class ProcessTextTest(unittest.TestCase):
  """ Can plain text be processed correctly? """

  def test(self):
    DTEULAAPP.loadmodels()
    results = DTEULAAPP.processClauseText(clausetext1, 'text')
    self.assertEqual(len(results), 18, 'There are 18 results of processing plain text')
    accs = sum(1 for curresult in results if curresult['classification'] == 'Acceptable')
    self.assertEqual(accs, 16, 'There are 16 acceptable clauses in the plain text')

class ProcessPDFTest(unittest.TestCase):
  """ Can PDF text be processed correctly? """

  def test(self):
    DTEULAAPP.loadmodels()
    resourcefolder = helpers.getresourcefolder()
    pdfPath = os.path.join(resourcefolder, 'sample_eula_1.pdf')
    curdata = ''
    with open(pdfPath, 'rb') as curfile:
      curdata = base64.b64encode(curfile.read()).decode()
    results = DTEULAAPP.processClauseText(curdata, 'pdf')
    self.assertEqual(len(results), 100, 'There are 100 results of processing PDF text')
    accs = sum(1 for curresult in results if curresult['classification'] == 'Acceptable')
    self.assertEqual(accs, 65, 'There are 65 acceptable clauses in the PDF text')

class ProcessWordTest(unittest.TestCase):
  """ Can Word text be processed correctly? """

  def test(self):
    DTEULAAPP.loadmodels()
    resourcefolder = helpers.getresourcefolder()
    wordPath = os.path.join(resourcefolder, 'sample_eula_1.docx')
    curdata = ''
    with open(wordPath, 'rb') as curfile:
      curdata = base64.b64encode(curfile.read()).decode()
    results = DTEULAAPP.processClauseText(curdata, 'word')
    self.assertEqual(len(results), 140, 'There are 140 results of processing Word text')
    accs = sum(1 for curresult in results if curresult['classification'] == 'Acceptable')
    self.assertEqual(accs, 139, 'There are 139 acceptable clauses in the Word text')

class ProcessNothingTest(unittest.TestCase):
  """ Can empty text be handled correctly? """

  def test(self):
    DTEULAAPP.loadmodels()
    results = DTEULAAPP.processClauseText('', 'text')
    self.assertEqual(len(results), 0, 'There are 0 results of processing no text')
    accs = sum(1 for curresult in results if curresult['classification'] == 'Acceptable')
    self.assertEqual(accs, 0, 'There are 0 acceptable clauses in the no text')

class ProcessArgleBargleTest(unittest.TestCase):
  """ Can Nonsense text be handled correctly? """

  def test(self):
    DTEULAAPP.loadmodels()
    results = DTEULAAPP.processClauseText(clausetext1, 'arglebargle')
    self.assertEqual(len(results), 0, 'There are 0 results of processing unknown format text')
    accs = sum(1 for curresult in results if curresult['classification'] == 'Acceptable')
    self.assertEqual(accs, 0, 'There are 0 acceptable clauses in the unknown format text')

class BatchTest(unittest.TestCase):
  """ Can batch processing work correctly? """

  def test(self):
    DTEULAAPP.loadmodels()
    resourcefolder = helpers.getresourcefolder()
    outpath = os.path.join(resourcefolder, 'tempout.json')
    DTEULAAPP.batch(resourcefolder, outpath)
    self.assertEqual(os.stat(outpath).st_size, 329185, 'The output file is 329185 bytes in size')
    os.remove(outpath)

if __name__ == '__main__':
  unittest.main()
