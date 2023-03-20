'use strict';
const axios = require('axios');
const striptags = require('striptags');
const _ = require('lodash');

const COMMENT_ANALYZER_URL =
  'https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze';
const MAX_LENGTH = 20480;

class PerspectiveAPIClientError extends Error {
  constructor(message) {
    super(message);
    Error.captureStackTrace(this, this.constructor);
    this.name = 'PerspectiveAPIClientError';
  }
}

class TextEmptyError extends PerspectiveAPIClientError {
  constructor() {
    super('text must not be empty');
    this.name = 'TextEmptyError';
  }
}

class TextTooLongError extends PerspectiveAPIClientError {
  constructor() {
    super(`text must not be greater than ${MAX_LENGTH} characters in length`);
    this.name = 'TextTooLongError';
  }
}

class ResponseError extends PerspectiveAPIClientError {
  constructor(message, response) {
    super(message);
    this.response = response;
    this.name = 'ResponseError';
  }
}

class Perspective {
  constructor(options) {
    this.options = options || {};
    if (!this.options.apiKey) {
      throw new Error('Must provide options.apiKey');
    }
  }

  analyze(text, options) {
    return new Promise((resolve, reject) => {
      let resource;
      try {
        resource = this.getAnalyzeCommentPayload(text, options);
      } catch (error) {
        reject(error);
      }
      axios
        .post(COMMENT_ANALYZER_URL, resource, {
          params: {key: this.options.apiKey},
        })
        .then(response => {
          resolve(response.data);
        }).catch(error => {
          const message = _.get(error, 'response.data.error.message', error.message);
          reject(new ResponseError(message, error.response));
        });
    });
  }

  getAnalyzeCommentPayload(text, options) {
    const opts = options || {};
    const stripHTML = opts.stripHTML == undefined ? true : opts.stripHTML;
    const truncate = opts.truncate == undefined ? false : opts.truncate;
    const doNotStore = opts.doNotStore == undefined ? true : opts.doNotStore;
    const validate = opts.validate == undefined ? true : opts.validate;
    const processText = str => {
      const ret = stripHTML ? striptags(str) : str;
      if (validate && !ret) {
        throw new TextEmptyError();
      }
      if (validate && !truncate && ret.length > MAX_LENGTH) {
        throw new TextTooLongError();
      }
      return truncate ? ret.substr(0, MAX_LENGTH) : ret;
    };
    let resource = {};
    if (typeof text === 'object') {
      resource = text;
      if (stripHTML && resource.comment.text) {
        resource.comment.text = processText(resource.comment.text);
      }
    } else {
      resource.comment = {text: processText(text)};
    }
    let attributes =
      opts.attributes == undefined && !resource.requestedAttributes
        ? {TOXICITY: {}}
        : opts.attributes;
    if (Array.isArray(opts.attributes)) {
      attributes = {};
      opts.attributes.forEach(each => {
        attributes[each.toUpperCase()] = {};
      });
    }
    return _.merge({}, resource, {
      requestedAttributes: attributes,
      doNotStore,
    });
  }
}

Perspective.PerspectiveAPIClientError = PerspectiveAPIClientError;
Perspective.TextEmptyError = TextEmptyError;
Perspective.TextTooLongError = TextTooLongError;
Perspective.ResponseError = ResponseError;
module.exports = Perspective;
