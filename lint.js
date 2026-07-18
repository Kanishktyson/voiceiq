const fs = require('fs');
const html = fs.readFileSync('index.html', 'utf8');
const start = html.indexOf('<script>') + 8;
const end = html.lastIndexOf('</script>');
const script = html.substring(start, end);
try {
    new Function(script);
    console.log('OK - no syntax errors');
} catch(e) {
    console.log('ERROR: ' + e.message);
    const m = e.stack.match(/eval:(\d+)/);
    if (m) console.log('Line: ' + m[1]);
}
