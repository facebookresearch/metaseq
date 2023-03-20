# perspective-api-client

[![Current Version](https://img.shields.io/npm/v/perspective-api-client.svg)](https://www.npmjs.org/package/perspective-api-client)
[![Build Status](https://travis-ci.org/sloria/perspective-api-client.svg?branch=master)](https://travis-ci.org/sloria/perspective-api-client)

Node.js client library for the [Perspective API](https://www.perspectiveapi.com/).

## Install

```
$ npm install perspective-api-client
```

## Usage

```js
const Perspective = require('perspective-api-client');
const perspective = new Perspective({apiKey: process.env.PERSPECTIVE_API_KEY});

(async () => {
  const text = 'you empty-headed animal food trough wiper!';
  const result = await perspective.analyze(text);
  console.log(JSON.stringify(result, null, 2));
})();
// {
//   "attributeScores": {
//     "TOXICITY": {
//       "spanScores": [
//         {
//           "begin": 0,
//           "end": 42,
//           "score": {
//             "value": 0.77587414,
//             "type": "PROBABILITY"
//           }
//         }
//       ],
//       "summaryScore": {
//         "value": 0.77587414,
//         "type": "PROBABILITY"
//       }
//     }
//   },
//   "languages": [
//     "en"
//   ]
// }
```

### Specifying models

The TOXICITY model is used by default. To specify additional models,
    pass `options.attributes`.

```js
(async () => {
  const text = 'fools!';
  const result = await perspective.analyze(text, {attributes: ['unsubstantial', 'spam']});
  console.log(JSON.stringify(result, null, 2));
})();
// {
//   "attributeScores": {
//     "UNSUBSTANTIAL": {
//       "spanScores": [
//         {
//           "begin": 0,
//           "end": 6,
//           "score": {
//             "value": 0.9592708,
//             "type": "PROBABILITY"
//           }
//         }
//       ],
//       "summaryScore": {
//         "value": 0.9592708,
//         "type": "PROBABILITY"
//       }
//     },
//     "SPAM": {
//       "spanScores": [
//         {
//           "begin": 0,
//           "end": 6,
//           "score": {
//             "value": 0.008744183,
//             "type": "PROBABILITY"
//           }
//         }
//       ],
//       "summaryScore": {
//         "value": 0.008744183,
//         "type": "PROBABILITY"
//       }
//     }
//   },
//   "languages": [
//     "en"
//   ]
// }
```

### More options

You can also pass an [AnalyzeComment](https://github.com/conversationai/perspectiveapi/blob/master/api_reference.md#analyzecomment-request)
object for more control over the request.

```js
(async () => {
  const text = 'you empty-headed animal food trough wiper!';
  const result = await perspective.analyze({
    comment: {text},
    requestedAttributes: {TOXICITY: {scoreThreshold: 0.7}},
  });
  console.log(JSON.stringify(result, null, 2));
})();
// {
//   "attributeScores": {
//     "TOXICITY": {
//       "spanScores": [
//         {
//           "begin": 0,
//           "end": 42,
//           "score": {
//             "value": 0.77587414,
//             "type": "PROBABILITY"
//           }
//         }
//       ],
//       "summaryScore": {
//         "value": 0.77587414,
//         "type": "PROBABILITY"
//       }
//     }
//   },
//   "languages": [
//     "en"
//   ]
// }
```

## API

### perspective = new Perspective()

#### analyze(text, [options])

#### text

Type: `String` or `Object`

Either the text to analyze or an [AnalyzeComment](https://github.com/conversationai/perspectiveapi/blob/master/api_reference.md#analyzecomment-request) object.
HTML tags will be stripped by default.

##### options

###### attributes

Type: `Array` or `Object`

Model names to analyze. `TOXICITY` is analyzed by default. If passing an Array of names, the names may be lowercased.
See https://github.com/conversationai/perspectiveapi/blob/master/api_reference.md#models
for a list of valid models.

###### doNotStore

Type: `Boolean`
Default: `true`

If `true`, prevent API from storing comment and context from this request.

##### stripHTML

Type: `Boolean`
Default: `true`

Whether to strip HTML tags from the text.

##### truncate

Type: `Boolean`
Default: `false`

If `true`, truncate text to the first 20480 characters (max length
    allowed by the Perspective API).

## FAQ

### How does this compare to @conversationai/perspectiveapi-js-client?

Similarities:

- Exposes the AnalyzeComment endpoint of the Perspective API
- Strips HTML tags by default

Differences:


- Returns full responses (rather than only returning summary scores)
- Exposes all [AnalyzeComment](https://github.com/conversationai/perspectiveapi/blob/master/api_reference.md#analyzecomment-request) options
- Supports all Node.js LTS versions

## License

MIT © [Steven Loria](http://stevenloria.com)
