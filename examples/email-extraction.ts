import { z } from "zod";
import { optimize, DefaultAdapterTask } from "../src";
import fs from "fs";
import path from "path";

// Minimal .env loader (no new dependencies)
function loadDotEnv(file = ".env") {
  const envPath = path.resolve(process.cwd(), file);
  if (!fs.existsSync(envPath)) return;
  const content = fs.readFileSync(envPath, "utf8");
  for (const raw of content.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const cleaned = line.startsWith("export ")
      ? line.slice("export ".length)
      : line;
    const eq = cleaned.indexOf("=");
    if (eq === -1) continue;
    const key = cleaned.slice(0, eq).trim();
    let val = cleaned.slice(eq + 1).trim();
    if (
      (val.startsWith('"') && val.endsWith('"')) ||
      (val.startsWith("'") && val.endsWith("'"))
    ) {
      val = val.slice(1, -1);
    }
    if (!(key in process.env)) {
      process.env[key] = val;
    }
  }
}

loadDotEnv();

if (!process.env.AI_GATEWAY_API_KEY && !process.env.VERCEL_OIDC_TOKEN) {
  console.warn(
    "Warning: AI_GATEWAY_API_KEY or VERCEL_OIDC_TOKEN not set. Add it to your .env file.",
  );
}

// Define the schema for extracting follow-up items from an email (array)
const Followup = z.object({
task: z.string().describe("Concise action item to follow up on"),
dueDate: z.string().optional().describe("Due date in YYYY-MM-DD if specified"),
priority: z.enum(["low", "medium", "high"]).describe("Priority of the follow-up"),
});
const Followups = z.array(Followup);

type FollowupT = z.infer<typeof Followup>;

// Example training data (each email -> array of follow-ups)
const trainset: DefaultAdapterTask<FollowupT[]>[] = [
{
input: `From: John Smith <john.smith@company.com>
Subject: Quarterly Report Due Friday

Hi team,

Just a reminder that the Q4 quarterly report is due this Friday, January 15th. 
Please ensure all your sections are completed and reviewed by EOD Thursday.

This is critical for our board meeting next week.

Best,
John`,
    expectedOutput: [
      { task: "Complete your section of the Q4 report", dueDate: "2024-01-14", priority: "high" },
      { task: "Submit final Q4 report", dueDate: "2024-01-15", priority: "high" },
    ],
    schema: Followups,
  },
  {
    input: `From: Marketing Team <marketing@company.com>
Subject: New Blog Post Published

Hello everyone,

We're excited to announce that our latest blog post "10 Tips for Remote Work" 
is now live on our website. Check it out and share with your networks!

Thanks,
Marketing Team`,
expectedOutput: [
  { task: "Share the blog post with your network", priority: "medium" },
],
  schema: Followups,
},
{
    input: `From: Sarah Johnson <sarah.j@client.com>
Subject: Re: Project Proposal - Need Clarification

Hi,

I've reviewed the project proposal you sent last week. Overall it looks good, 
but I need some clarification on the timeline and budget sections. 

Can we schedule a call tomorrow afternoon to discuss? I'm available 2-4 PM EST.

Best regards,
Sarah`,
expectedOutput: [
{ task: "Schedule a call with Sarah (2–4 PM EST)", priority: "high" },
{ task: "Prepare clarifications for timeline and budget", priority: "medium" },
],
schema: Followups,
},
];

// Scorer for arrays of follow-ups (best-match + penalty for extras)
function pairScore(a: FollowupT, b: FollowupT): number {
  const taskMatch = a.task.toLowerCase().includes(b.task.toLowerCase()) || b.task.toLowerCase().includes(a.task.toLowerCase());
  const dueMatch = (a.dueDate || "") === (b.dueDate || "");
  const prioMatch = a.priority === b.priority;
  let s = 0;
  if (taskMatch) s += 0.6;
  if (prioMatch) s += 0.3;
  if (dueMatch && b.dueDate) s += 0.1;
  return s;
}

function followupsScorer(pred: FollowupT[], expected?: FollowupT[]): number {
if (!expected) return 1;
if (!Array.isArray(pred)) return 0;
if (expected.length === 0) return pred.length === 0 ? 1 : 0;
const perExp = expected.map((e) => {
let best = 0;
for (const p of pred) best = Math.max(best, pairScore(p, e));
return best;
});
const base = perExp.reduce((a, b) => a + b, 0) / perExp.length;
const penalty = expected.length / Math.max(expected.length, pred.length);
  return Math.max(0, Math.min(1, base * penalty));
}

// Add scorer to tasks
const tasksWithScorer = trainset.map((task) => ({
...task,
scorer: followupsScorer,
}));

async function runExample() {
  console.log("Starting GEPA optimization for email extraction...\n");

  const startingPrompt = `You are an email follow-up extractor.
  
  From the email body, extract an ARRAY of follow-up items (0–5). Each item MUST be a JSON object with keys:
  - task: concise imperative action to perform (e.g., "Schedule a call with Sarah").
  - dueDate (optional): primary deadline in ISO YYYY-MM-DD if stated clearly.
  - priority: one of "high", "medium", "low".
  
  Priority guidance:
  - high: explicit urgency or deadlines (e.g., reminder, due Friday, urgent, today, EOD).
  - medium: time-bound but not urgent (e.g., this week, tomorrow, schedule a meeting).
  - low: informational or general suggestions.
  
  Rules:
  - Do not fabricate details—only extract actions implied or explicitly requested by the sender.
  - If multiple deadlines appear, pick the earliest clear date for dueDate.
  - Omit dueDate when the email does not specify any date.
  - Return strictly valid JSON (an array) with double quotes, no comments, and no extra text.`;

  const t0 = Date.now();
  const result = await optimize({
    seedCandidate: {
      system: startingPrompt,
    },
    trainset: tasksWithScorer,
    valset: tasksWithScorer, // In practice, use separate validation set
    taskLM: { model: "openai/gpt-5-nano", maxConcurrency: 10 }, // Gateway model id
    reflectionLM: "openai/o3", // Stronger model for reflection (Gateway model id)
    maxIterations: 5,
    maxMetricCalls: 100,
    maxBudgetUSD: 50,
    reflectionMinibatchSize: 2,
    candidateSelectionStrategy: "pareto",
    componentSelector: "round_robin",
    logger: {
      log: (level, message, data) => {
        if (level === "info") {
          console.log(`[${level}] ${message}`, data || "");
        }
      },
    },
    persistence: {
      dir: "runs/email",
      checkpointEveryIterations: 1,
      resume: true,
    },
  });
  const durationMs = Date.now() - t0;

  console.log("\n=== Optimization Results ===");
  console.log("Best Score:", result.bestScore);
  console.log("Total Iterations:", result.iterations);
  console.log("Total Metric Calls:", result.totalMetricCalls);
  console.log("Total Cost (USD):", result.totalCostUSD?.toFixed(4));
  console.log("Total Duration (ms):", durationMs);
  console.log("\nBest System Prompt:");
  console.log(result.bestCandidate.system);

  // Show improvement over iterations
  console.log("\n=== Iteration History ===");
  result.history.slice(-5).forEach((h) => {
    console.log(
      `Iteration ${h.iteration}: ${h.accepted ? "Accepted" : "Rejected"}`,
    );
  });
}

// Note: To run this example, you need:
// 1. Create .env with AI_GATEWAY_API_KEY=...
// 2. Run: npm run example

if (require.main === module) {
  runExample().catch(console.error);
}
