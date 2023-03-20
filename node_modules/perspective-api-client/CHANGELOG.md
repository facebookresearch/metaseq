# Changelog

### 3.1.0 (2018-12-11)

- Increase max text length to 20480 characters.
- Drop support for node 4.

## 3.0.0 (2017-12-08)

- A `ResponseError` is thrown if the API returns an error.

## 2.0.0 (2017-12-03)

- Directly call to comment analyzer API rather than using Google's
discovery API to dynamically generate a client. This saves an API
request when doing an analysis.
- Validate text before it gets sent to the API.
- Rename `makeResource` to `getAnalyzeCommentPayload`.

## 1.1.0 (2017-12-02)

- Add `truncate` option.

## 1.0.0 (2017-12-02)

- First stable release.
