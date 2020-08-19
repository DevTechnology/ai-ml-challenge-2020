# -*- coding: utf-8 -*-
"""EULA app API endpoints and index page"""

# Standard library imports
import logging
import os
import pickle
import sys

# Third-party imports
import flask
from flask.wrappers import Response
from flask import Blueprint

# hts imports
import dteulaapp.core as EULACORE

landingpage = Blueprint('landingpage', __name__, template_folder='templates')
theapi = Blueprint('theapi', __name__, template_folder='templates')
userfeedback = Blueprint('feedback', __name__, template_folder='templates')

@landingpage.route('/')
def index(): # type: () -> str
  """
  Publish the index.html from resources
  """

  return flask.render_template('index.html')

@theapi.route('/clause', methods=['POST'])
def clause(): # type: () -> Response
  """
  Handle clause input from the page.  This may be raw text pasted into a text field
  or base64-encoded binary data in PDF or Word format.

  :return: A Flask Response with type JSON containing the clause processing results.
  :rtype: Response
  """

  if flask.request.method == 'POST':
    reqdict = flask.request.get_json(force=True)
    clauseinput = reqdict['clausetext']
    mtype = reqdict['mtype']
    results = EULACORE.processClauseText(clauseinput, mtype)
    return flask.jsonify(results)

  return flask.jsonify([])

@userfeedback.route('/feedback', methods=['POST'])
def feedback(): # type: () -> Response
  """
  Handle feedback from the user.

  :return: A Flask Response with acknowledgement that it went OK
  :rtype: Response
  """

  if flask.request.method == 'POST':
    reqdict = flask.request.get_json(force=True)
    EULACORE.dealwithfeedback(reqdict)
    return flask.jsonify({'message': 'feedback acknowledged'})

  return flask.jsonify({'message': 'that did not look right'})
