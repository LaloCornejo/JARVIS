const path = require('path');
const Module = require('module');
const EventEmitter = require('events');

const events = new EventEmitter();

const emit = (request, id) => {
  var type = 'file';

  if (id.indexOf('node_modules') > -1) {
    type = 'module';
  } else if (!path.parse(id).root) {
    type = 'builtin';
  }

  events.emit('file', {
    type: type,
    id: id
  });
};

// this is a hack and I don't like it
// if you know a better way to collect all required modules
// please file an issue, I'd appreciate it very much
// https://github.com/catdad/electronmon/issues/new
const originalLoad = Module._load;
Module._load = function (request) {
  const filename = Module._resolveFilename.apply(this, arguments);

  if (filename && !require.cache[filename]) {
    emit(request, filename);
  }

  return originalLoad.apply(this, arguments);
};

module.exports = events;
