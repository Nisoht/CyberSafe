
import { serve } from "https://deno.land/std@0.190.0/http/server.ts";
import { Resend } from "npm:resend@2.0.0";

const resend = new Resend(Deno.env.get("RESEND_API_KEY"));

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

interface WelcomeEmailRequest {
  email: string;
  username: string;
}

const handler = async (req: Request): Promise<Response> => {
  // Handle CORS preflight requests
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    console.log("Received request to send welcome email");
    
    if (req.method !== "POST") {
      throw new Error("Method not allowed");
    }
    
    const body = await req.text();
    console.log("Request body:", body);
    
    let data: WelcomeEmailRequest;
    try {
      data = JSON.parse(body);
    } catch (e) {
      console.error("Error parsing JSON:", e);
      throw new Error("Invalid JSON");
    }
    
    const { email, username } = data;

    if (!email) {
      throw new Error("Email is required");
    }

    console.log("Sending welcome email to:", email, "with username:", username || "there");

    const emailResponse = await resend.emails.send({
      from: "CyberSafe <onboarding@resend.dev>",
      to: [email],
      subject: "Welcome to CyberSafe!",
      html: `
        <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px;">
          <h1 style="color: #4f46e5; margin-bottom: 20px;">Welcome to CyberSafe!</h1>
          <p>Hello ${username || "there"},</p>
          <p>Thank you for creating an account with CyberSafe. We're excited to have you join our community dedicated to making online spaces safer for everyone.</p>
          <p>With your CyberSafe account, you can:</p>
          <ul>
            <li>Analyze text for potential cyberbullying</li>
            <li>Get insights on harmful content</li>
            <li>Help create a safer digital environment</li>
          </ul>
          <p>If you have any questions or need assistance, don't hesitate to reach out to our support team.</p>
          <p style="margin-top: 30px;">Best regards,</p>
          <p>The CyberSafe Team</p>
        </div>
      `,
    });

    console.log("Welcome email sent successfully:", emailResponse);

    return new Response(JSON.stringify({ success: true }), {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        ...corsHeaders,
      },
    });
  } catch (error: any) {
    console.error("Error sending welcome email:", error);
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        status: 500,
        headers: { "Content-Type": "application/json", ...corsHeaders },
      }
    );
  }
};

serve(handler);
