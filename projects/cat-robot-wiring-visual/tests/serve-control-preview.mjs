import {createServer} from 'node:http';
import {readFileSync} from 'node:fs';
import {createFixture} from './control-fixture.mjs';

// This server is a browser fixture. It never opens a serial port.
const source = readFileSync(new URL('../sketches/cat_robot/control_page.h', import.meta.url), 'utf8');
const page = source.split('R"HTML(')[1].split(')HTML";')[0].replace('__TOKEN__', 'preview-only');
const fixture = createFixture();
createServer(async (request, response) => {
  let body = '';
  for await (const chunk of request) body += chunk;
  response.setHeader('Cache-Control', 'no-store');
  if (request.url === '/') {response.setHeader('Content-Type', 'text/html; charset=utf-8'); response.end(page); return;}
  response.setHeader('Content-Type', 'application/json');
  const result = await fixture(request.url, {method: request.method, body,
    headers: {'X-Robot-Token': request.headers['x-robot-token']}});
  response.statusCode = result.status;
  response.end(JSON.stringify(await result.json()));
}).listen(8766, '127.0.0.1', () => console.log('Preview fixture: http://127.0.0.1:8766/ (no hardware)'));
