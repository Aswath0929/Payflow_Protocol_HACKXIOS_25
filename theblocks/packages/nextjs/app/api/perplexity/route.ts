import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    // Handle empty or malformed requests
    let body;
    try {
      const text = await request.text();
      if (!text || text.trim() === '') {
        return NextResponse.json({ error: "Empty request body" }, { status: 400 });
      }
      body = JSON.parse(text);
    } catch (parseError) {
      console.error("JSON Parse Error:", parseError);
      return NextResponse.json({ error: "Invalid JSON in request body" }, { status: 400 });
    }
    
    const apiKey = "process.env.PERPLEXITY_API_KEY";

    const response = await fetch("https://api.perplexity.ai/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${apiKey}`
      },
      body: JSON.stringify(body)
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error("Perplexity API Error:", response.status, errorText);
      return NextResponse.json({ error: `Perplexity API error: ${response.status}`, details: errorText }, { status: response.status });
    }

    // If streaming is requested, return the stream
    if (body.stream) {
      return new Response(response.body, {
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
        },
      });
    }

    // Otherwise return JSON
    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Proxy Error:", error);
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}

