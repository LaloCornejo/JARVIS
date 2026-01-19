# runtime-required

[![travis][travis.svg]][travis.link]
[![score-codeclimate][score-codeclimate.svg]][score-codeclimate.link]
[![npm-downloads][npm-downloads.svg]][npm.link]
[![npm-version][npm-version.svg]][npm.link]
[![dm-david][dm-david.svg]][dm-david.link]

[travis.svg]: https://travis-ci.com/catdad/runtime-required.svg?branch=master
[travis.link]: https://travis-ci.com/catdad/runtime-required
[score-codeclimate.svg]: https://codeclimate.com/github/catdad/runtime-required/badges/gpa.svg
[score-codeclimate.link]: https://codeclimate.com/github/catdad/runtime-required
[npm-downloads.svg]: https://img.shields.io/npm/dm/runtime-required.svg
[npm.link]: https://www.npmjs.com/package/runtime-required
[npm-version.svg]: https://img.shields.io/npm/v/runtime-required.svg
[dm-david.svg]: https://david-dm.org/catdad/runtime-required.svg
[dm-david.link]: https://david-dm.org/catdad/runtime-required

Haven't you ever always needed to know what files are being required by your app? Okay, maybe not always, but once? No? Just me? Fine, move along then. But in case you need it, here's how it all works:

## Install

It's in npm, of course:

```bash
npm install runtime-required
```

## Usage

### `require('runtime-required')` → [`EventEmitter`](https://nodejs.org/api/events.html)

Yup, you just include it in your project and it exposes an event emitter. There is only a single `file` event, and it has the following properties:
* **`type`** _{string}_: one of
  * `builtin` (_node's default modules_)
  * `module` (_ones that appear inside the `node_modules` directory_)
  * `file` (_from your own project... or I guess elsewhere on the filesystem_)
* **`id`** _{string}_: the identifier for this module. For `module` and `file` types, it will be a fully-resolved file path. For `builtin` types, it will be the name of the module.

## Example

```javascript
const required = require('runtime-required');

required.on('file', data => {
  console.log(`a ${data.type} module was required at "${data.id}"`);
});
```
