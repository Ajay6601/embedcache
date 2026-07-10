// Animated terminal: plays a script of commands and output into a <pre>,
// typing commands character by character, then loops.
function playTerminal(preId, script, opts) {
  const pre = document.getElementById(preId);
  if (!pre) return;
  const typeMs = (opts && opts.typeMs) || 26;
  const lineMs = (opts && opts.lineMs) || 60;
  const holdMs = (opts && opts.holdMs) || 5200;
  let html = "";
  const cursor = '<span class="cursor"></span>';
  const render = (tail) => { pre.innerHTML = html + (tail || ""); };

  let started = false;
  const io = new IntersectionObserver((entries) => {
    if (entries.some(e => e.isIntersecting) && !started) { started = true; step(0); }
  }, { threshold: 0.2 });
  io.observe(pre);

  function step(i) {
    if (i >= script.length) {
      render(cursor);
      setTimeout(() => { html = ""; step(0); }, holdMs);
      return;
    }
    const item = script[i];
    if (item.cmd !== undefined) {
      const prompt = '<span class="c-prompt">$ </span>';
      let j = 0;
      const typeNext = () => {
        render(prompt + '<span class="c-cmd">' + esc(item.cmd.slice(0, j)) + '</span>' + cursor);
        if (j++ < item.cmd.length) { setTimeout(typeNext, typeMs); }
        else {
          html += prompt + '<span class="c-cmd">' + esc(item.cmd) + '</span>\n';
          setTimeout(() => step(i + 1), item.pause || 350);
        }
      };
      typeNext();
    } else {
      html += item.html !== undefined ? item.html + "\n" : esc(item.out) + "\n";
      render(cursor);
      setTimeout(() => step(i + 1), item.pause || lineMs);
    }
  }
  function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;"); }
}

// copy buttons for code blocks
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("pre.code").forEach((pre) => {
    const btn = document.createElement("button");
    btn.className = "copy";
    btn.textContent = "copy";
    btn.addEventListener("click", () => {
      navigator.clipboard.writeText(pre.innerText.replace(/^copy\n?/, "").trim());
      btn.textContent = "copied!";
      setTimeout(() => (btn.textContent = "copy"), 1400);
    });
    pre.appendChild(btn);
  });
  const inst = document.getElementById("install-cmd");
  if (inst) {
    inst.addEventListener("click", () => {
      navigator.clipboard.writeText(inst.querySelector("code").innerText);
      const c = inst.querySelector(".copied");
      if (c) { c.textContent = "copied!"; setTimeout(() => (c.textContent = "click to copy"), 1400); }
    });
  }
});
