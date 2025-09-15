import { z } from "zod";
import { optimize, DefaultAdapterTask } from "../src";
import fs from "fs";
import path from "path";

function loadDotEnv(file = ".env") {
  const p = path.resolve(process.cwd(), file);
  if (!fs.existsSync(p)) return;
  for (const raw of fs.readFileSync(p, "utf8").split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#")) continue;
    const cleaned = line.startsWith("export ") ? line.slice(7) : line;
    const eq = cleaned.indexOf("=");
    if (eq === -1) continue;
    const k = cleaned.slice(0, eq).trim();
    let v = cleaned.slice(eq + 1).trim();
    if ((v.startsWith('"') && v.endsWith('"')) || (v.startsWith("'") && v.endsWith("'"))) v = v.slice(1, -1);
    if (!(k in process.env)) process.env[k] = v;
  }
}

loadDotEnv();

// Classification schema for content safety + spam detection
const MessageClass = z.object({
  harmful: z.boolean().describe("Whether the message contains harmful/abusive/unsafe content."),
  spam: z.boolean().describe("Whether the message is unsolicited promotional or deceptive spam."),
  rationale: z.string().optional().describe("Brief reason for the decision."),
});

type MsgCls = z.infer<typeof MessageClass>;

// Helper to build tasks
function task(input: string, harmful: boolean, spam: boolean): DefaultAdapterTask<MsgCls> {
  return { input, expectedOutput: { harmful, spam, rationale: "" }, schema: MessageClass };
}

// Train examples
const trainset: DefaultAdapterTask<MsgCls>[] = [
  task("Hey team, the build passed and I've deployed to staging. Please review the changelog.", false, false),
  task("WIN A FREE iPhone!!! Click here now: http://scam.example.com", false, true),
  task("You're an idiot and don't deserve this job.", true, false),
  task("URGENT: Your bank account was compromised. Verify now at http://phish.example.com", true, true),
  task("Big sale on shoes this weekend! Use code SAVE20 at checkout.", false, true),
  task("Let's schedule the design review for Tuesday 2pm.", false, false),
  task("I hope you fail your exam and get expelled.", true, false),
  task("Limited time crypto investment opportunity. 10x returns guaranteed!!!", false, true),
  task("Could you please share the Q3 metrics doc?", false, false),
  task("Kill yourself.", true, false),
];

// Validation examples
const valset: DefaultAdapterTask<MsgCls>[] = [
  task("Reminder: standup at 10am.", false, false),
  task("Buy followers now! Best prices: cheapfollowers.example", false, true),
  task("Shut up, nobody cares about your opinion.", true, false),
  task("Final notice: your package is on hold, pay the customs fee here: http://fakepay.example", true, true),
];

function accuracyBoth(pred: MsgCls, expected?: MsgCls): number {
  if (!expected) return 1;
  let score = 0;
  if (pred?.harmful === expected.harmful) score += 0.5;
  if (pred?.spam === expected.spam) score += 0.5;
  return score;
}

const trainWithScorer = trainset.map((t) => ({ ...t, scorer: accuracyBoth }));
const valWithScorer = valset.map((t) => ({ ...t, scorer: accuracyBoth }));

async function run() {
  console.log("Starting GEPA optimization for message harm/spam classification...\n");
  const t0 = Date.now();
  const result = await optimize({
    seedCandidate: {
      system:
        "You are a content policy classifier. Given a message (a single string), output JSON {harmful: boolean, spam: boolean, rationale?}.\n- harmful=true if the content is abusive, threatening, inciting, self-harm, or otherwise unsafe.\n- spam=true if it is unsolicited promotional or deceptive content (scams, phishing, link/SEO spam).\nKeep rationale concise.",
    },
    trainset: trainWithScorer,
    valset: valWithScorer,
    taskLM: "openai/gpt-5-nano",
    reflectionLM: "openai/o3",
    maxIterations: 6,
    maxMetricCalls: 200,
    maxBudgetUSD: 50,
    reflectionMinibatchSize: 3,
    candidateSelectionStrategy: "pareto",
    componentSelector: "round_robin",
    logger: { log: (lvl, msg, data) => { if (lvl === "info") console.log(`[${lvl}] ${msg}`, data || ""); } },
    persistence: { dir: "runs/message-spam", checkpointEveryIterations: 1, resume: true },
  });
  const durationMs = Date.now() - t0;

  console.log("\n=== Optimization Results (Message Harm/Spam) ===");
  console.log("Best Score:", result.bestScore);
  console.log("Total Iterations:", result.iterations);
  console.log("Total Metric Calls:", result.totalMetricCalls);
  console.log("Total Cost (USD):", result.totalCostUSD?.toFixed(4));
  console.log("Total Duration (ms):", durationMs);
  console.log("\nBest System Prompt:\n" + result.bestCandidate.system);
}

if (require.main === module) {
  run().catch((err) => { console.error(err); process.exit(1); });
}
