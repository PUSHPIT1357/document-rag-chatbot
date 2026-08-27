import { createStartHandler, defaultStreamHandler } from "@tanstack/react-start/server";
import { consumeLastCapturedError } from "./lib/error-capture";
import { renderErrorPage } from "./lib/error-page";

const handler = createStartHandler(defaultStreamHandler);

async function fetch(request: Request): Promise<Response> {
  try {
    return await handler(request);
  } catch (error) {
    const original = consumeLastCapturedError() ?? error;
    console.error("SSR error:", original);
    return new Response(renderErrorPage(), {
      status: 500,
      headers: { "Content-Type": "text/html" },
    });
  }
}

export default { fetch };