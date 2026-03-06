/*
  EimemesChat AI — Express backend
  Model: HuggingFace Inference API (meta-llama/Llama-3.1-8B-Instruct)
  Hard 30s timeout. On failure, returns a friendly AI-style error message.
*/

'use strict';
require('dotenv').config();

const express = require('express');
const path    = require('path');
const app     = express();

app.use(express.json({ limit: '1mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// ── Logger ──────────────────────────────────────────────────────
const ts  = () => new Date().toISOString();
const log = (tag, msg) => console.log(`[${ts()}] [${tag}] ${msg}`);

// ── Timeout helper ──────────────────────────────────────────────
function withTimeout(promise, ms, label) {
  let id;
  const timer = new Promise((_, reject) => {
    id = setTimeout(
      () => reject(new Error(`${label} timed out after ${ms / 1000}s`)),
      ms
    );
  });
  return Promise.race([promise, timer]).finally(() => clearTimeout(id));
}

const PROVIDER_TIMEOUT = 30_000; // 30 s

// ── HuggingFace provider ────────────────────────────────────────
async function tryHuggingFace(messages, abortSignal) {
  if (!process.env.HUGGINGFACE_API_KEY) {
    throw new Error('HUGGINGFACE_API_KEY is not configured on the server.');
  }

  const fetchCall = fetch(
    'https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct/v1/chat/completions',
    {
      method: 'POST',
      signal: abortSignal,
      headers: {
        'Content-Type': 'application/json',
        Authorization:  `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
      },
      body: JSON.stringify({
        model:      'meta-llama/Llama-3.1-8B-Instruct',
        max_tokens: 1024,
        messages: [
          {
            role:    'system',
            content: 'You are Eimemes AI, a helpful, knowledgeable, and friendly assistant. Keep responses clear and well-formatted. Be concise unless detail is explicitly requested.',
          },
          ...messages.map(m => ({
            role:    m.role === 'assistant' ? 'assistant' : 'user',
            content: m.content,
          })),
        ],
      }),
    }
  );

  const res = await withTimeout(fetchCall, PROVIDER_TIMEOUT, 'HuggingFace');

  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`HuggingFace HTTP ${res.status}: ${body.slice(0, 200)}`);
  }

  const data = await res.json();
  const reply = data?.choices?.[0]?.message?.content;
  if (!reply) throw new Error('HuggingFace: unexpected response shape');

  return { reply, model: 'Llama 3.1' };
}

// ── Friendly error messages returned AS AI replies ──────────────
function buildErrorReply(err) {
  if (err.message.includes('timed out')) {
    return "I'm sorry, I took too long to respond and the request timed out. Please try sending your message again — I'll be quicker this time!";
  }
  if (err.message.includes('HUGGINGFACE_API_KEY')) {
    return "I'm currently unavailable because the AI service hasn't been configured on the server. Please contact the administrator to set up the API key.";
  }
  if (err.message.includes('HTTP 429') || err.message.includes('rate limit')) {
    return "I'm getting a lot of messages right now and have hit a rate limit. Please wait a moment and try again — I'll be back shortly!";
  }
  if (err.message.includes('HTTP 503') || err.message.includes('loading')) {
    return "The AI model is currently loading or warming up. This can take 20–30 seconds on first use. Please try again in a moment!";
  }
  return "I ran into an unexpected error and couldn't respond. Please try again — if this keeps happening, the service may be temporarily down.";
}

// ── POST /api/chat ───────────────────────────────────────────────
app.post('/api/chat', async (req, res) => {
  const { message, history = [] } = req.body;

  if (!message?.trim()) {
    return res.status(400).json({ error: 'Message is required.' });
  }

  // Build message array: history + current user turn
  const messages = [
    ...history
      .filter(m => m.role && m.content)
      .map(m => ({ role: m.role, content: m.content })),
    { role: 'user', content: message.trim() },
  ];

  // Forward abort when client disconnects
  const ctrl = new AbortController();
  req.on('close', () => {
    if (!res.headersSent) ctrl.abort();
  });

  try {
    log('AI', 'Calling HuggingFace…');
    const result = await tryHuggingFace(messages, ctrl.signal);
    log('AI', `✓ HuggingFace responded (${result.reply.length} chars)`);
    return res.json(result);

  } catch (err) {
    // Client disconnected — do nothing
    if (err.name === 'AbortError' || ctrl.signal.aborted) {
      log('AI', 'Client disconnected — aborting');
      return;
    }

    log('AI', `✗ HuggingFace failed — ${err.message}`);

    // Return a friendly AI-style reply so the chat always shows something
    return res.json({
      reply:  buildErrorReply(err),
      model:  'error',
      isError: true,
    });
  }
});

// ── GET /api/health ──────────────────────────────────────────────
app.get('/api/health', (_req, res) => {
  res.json({
    status:    'ok',
    provider:  'HuggingFace',
    model:     'meta-llama/Llama-3.1-8B-Instruct',
    configured: !!process.env.HUGGINGFACE_API_KEY,
    timestamp: new Date().toISOString(),
  });
});

// ── Catch-all: serve index.html ──────────────────────────────────
app.get('*', (_req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ── Start ────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  log('SERVER', `EimemesChat running → http://localhost:${PORT}`);
  if (process.env.HUGGINGFACE_API_KEY) {
    log('SERVER', '✓ HUGGINGFACE_API_KEY is set — ready to serve requests');
  } else {
    log('SERVER', '⚠️  HUGGINGFACE_API_KEY not set — add it to .env');
  }
});
