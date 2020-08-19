# -*- coding: utf-8 -*-
""" dteulaapp module

"""
# Standard library imports
import logging
from logging.handlers import RotatingFileHandler
from logging import Formatter
import os

# Third-party imports
import flask

__version__ = "0.0.1"

def create_app(): # type: () -> flask.Flask
  """
  Application context initialization.  This method MUST
  have this name.

  :return: the contextualized flask app
  :rtype: flask.Flask
  """

  app = flask.Flask(__name__, instance_relative_config=False)
  with app.app_context():
    from . import core
    from . import eulaapp

    core.loadmodels()
    app.register_blueprint(eulaapp.landingpage)
    app.register_blueprint(eulaapp.theapi)
    app.register_blueprint(eulaapp.userfeedback)

    return app
