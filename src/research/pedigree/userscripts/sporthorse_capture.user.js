// ==UserScript==
// @name         Dormello DOB capture (SporthorseData -> ローカル入力ツール)
// @namespace    keiba-vpn
// @version      1.1
// @description  ご自身が開いた SporthorseData の馬ページから Date of Birth を読み取り、127.0.0.1 のローカル入力ツールへ送る（自動遷移・自動アクセスはしない）
// @match        https://sporthorse-data.com/pedigree/*
// @grant        GM_xmlhttpRequest
// @connect      127.0.0.1
// ==/UserScript==
(function () {
  'use strict';
  const BASE = 'http://127.0.0.1:8765';
  const text = document.body ? document.body.innerText : '';
  // 例: "Date of Birth: 30-Jan-1970" / "born 30-Jan-1970"。年のみのときは日付なし扱い
  const full = text.match(/(?:Date of Birth|born)[:\s]*([0-9]{1,2}-[A-Za-z]{3}-[0-9]{4})/i);
  const yearOnly = !full && text.match(/(?:Date of Birth|born)[:\s]*([0-9]{4})\b/i);

  function call(method, path, body, token) {
    return new Promise((resolve, reject) => {
      GM_xmlhttpRequest({
        method, url: BASE + path,
        headers: Object.assign({ 'Content-Type': 'application/json' }, token ? { 'X-Token': token } : {}),
        data: body ? JSON.stringify(body) : undefined,
        onload: r => { try { resolve(JSON.parse(r.responseText)); } catch (e) { reject(e); } },
        onerror: () => reject(new Error('ローカル入力ツールに接続できません（起動していますか）')),
      });
    });
  }
  async function send(path, body) {
    const t = (await call('GET', '/api/token')).token;
    return call('POST', path, Object.assign({ url: location.href, title: document.title }, body), t);
  }

  const inFrame = window.top !== window.self;   // ツールの画面に埋め込まれているとき
  // 読み取った日付はツールの画面へ送る（保存は画面のボタンで行う）。年のみのときも知らせる
  const report = async () => {
    try { await send('/api/ext_detect', { text: full ? full[1] : null, year_only: yearOnly ? yearOnly[1] : null }); } catch (e) { /* ツールが起動していないときは何もしない */ }
  };
  report();
  if (inFrame) return;                              // 埋め込み中は表示を増やさない
  const box = document.createElement('div');
  box.style.cssText = 'position:fixed;right:12px;bottom:12px;z-index:2147483647;background:#fffbe6;border:2px solid #d4a800;border-radius:8px;padding:10px 12px;font:14px sans-serif;color:#000;box-shadow:0 2px 8px rgba(0,0,0,.3);max-width:340px';
  const msg = document.createElement('div');
  const input = document.createElement('input');
  input.style.cssText = 'width:150px;margin:6px 6px 0 0;padding:4px';
  const row = document.createElement('div');
  box.append(msg, input, row);
  function btn(label, fn) {
    const b = document.createElement('button');
    b.textContent = label;
    b.style.cssText = 'margin:6px 6px 0 0;padding:5px 10px;cursor:pointer';
    b.onclick = fn;
    row.append(b);
    return b;
  }
  async function run(path, body) {
    msg.textContent = '送信中…';
    try {
      const r = await send(path, body);
      msg.textContent = r.ok ? '保存しました。ツールの画面へ戻ってください' : ('保存できません: ' + r.error);
      msg.style.color = r.ok ? '#0a7d2c' : '#b00020';
    } catch (e) { msg.textContent = String(e.message || e); msg.style.color = '#b00020'; }
  }
  if (full) {
    input.value = full[1];
    msg.innerHTML = '検出した誕生日: <b>' + full[1] + '</b>';
    btn('この日付で保存', () => run('/api/ext_capture', { text: input.value }));
  } else {
    msg.textContent = yearOnly ? 'このページは年のみ（' + yearOnly[1] + '）のようです' : '誕生日を検出できませんでした。ページを見て手で入力できます';
    input.placeholder = '例 30-Jan-1970';
    btn('入力した日付で保存', () => run('/api/ext_capture', { text: input.value }));
  }
  btn('記載なし', () => run('/api/ext_none', {}));
  document.body.appendChild(box);
})();
