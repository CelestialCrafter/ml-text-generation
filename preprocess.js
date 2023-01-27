const fs = require('fs');
let dataset = require('./dataset.json');

dataset = dataset.messages.map(message => `${message.author.name}: ${message.content}`);
// eslint-disable-next-line max-len, no-useless-escape
dataset = dataset.map(message => message.replace(/((([A-Za-z]{3,9}:(?:\/\/)?)(?:[\-;:&=\+\$,\w]+@)?[A-Za-z0-9\.\-]+|(?:www\.|[\-;:&=\+\$,\w]+@)[A-Za-z0-9\.\-]+)((?:\/[\+~%\/\.\w\-_]*)?\??(?:[\-\+=&;%@\.\w_]*)#?(?:[\.\!\/\\\w]*))?)/g, '').replace(/[^\s\w!@()\-=_.?<>'";:|]/g, '')).filter(message => !(message === ''));

fs.writeFileSync('dataset.txt', dataset.join('\n'));
