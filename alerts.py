"""
HTML email alert sender via SMTP.
Reads credentials from config (which reads from st.secrets or .env).
"""
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone

import config

logger = logging.getLogger(__name__)


def _html_email(
    symbol: str, name: str, price: float, change_pct: float,
    confidence: float, score: float, direction: str,
    catalysts: list[str], signals: list[dict],
    stop_loss=None, take_profit_1=None, take_profit_2=None,
    support=None, resistance=None, risk_reward=None, position_pct=None,
) -> str:
    c        = "#00C805" if direction == "bullish" else "#F23645" if direction == "bearish" else "#94a3b8"
    arrow    = "▲" if direction == "bullish" else "▼" if direction == "bearish" else "▶"
    chg_c    = "#00C805" if change_pct >= 0 else "#F23645"
    chg_str  = f"+{change_pct:.2f}%" if change_pct >= 0 else f"{change_pct:.2f}%"
    ts       = datetime.now(timezone.utc).strftime("%b %d, %Y at %H:%M UTC")

    cat_label = "CATALYSTS" if direction == "bullish" else "RISKS"
    cat_li   = "".join(f"<li style='margin-bottom:6px;color:#94a3b8;'>{x}</li>" for x in catalysts)
    sig_span = "".join(
        f"""<span style='display:inline-block;margin:3px;padding:3px 10px;border-radius:20px;font-size:12px;
            font-weight:600;background:{"rgba(0,200,5,0.15)" if s["type"]=="bullish" else "rgba(242,54,69,0.15)"};
            color:{"#00C805" if s["type"]=="bullish" else "#F23645"};
            border:1px solid {"rgba(0,200,5,0.4)" if s["type"]=="bullish" else "rgba(242,54,69,0.4)"};'>{s["name"]}</span>"""
        for s in signals
    )

    profit_html = ""
    if stop_loss is not None and take_profit_1 is not None:
        sup = f"${support:.2f}" if support is not None else "—"
        res = f"${resistance:.2f}" if resistance is not None else "—"
        rr = f"1:{risk_reward:.1f}" if risk_reward is not None else "—"
        pos = f"{position_pct:.1f}%" if position_pct is not None else "—"
        profit_html = f"""
  <tr><td style="padding:20px 32px;border-bottom:1px solid #1c2840;background:#0a0e18;">
    <div style="font-size:11px;font-weight:700;letter-spacing:2px;color:#00b4d8;margin-bottom:12px;">📊 TRADE PLAN (ATR-based)</div>
    <table width="100%" style="font-size:13px;"><tr>
      <td><span style="color:#64748b;">Stop-loss:</span> <span style="color:#F23645;font-weight:600;">${stop_loss:.2f}</span></td>
      <td><span style="color:#64748b;">Take-profit 1:</span> <span style="color:#00C805;font-weight:600;">${take_profit_1:.2f}</span></td>
      <td><span style="color:#64748b;">Take-profit 2:</span> <span style="color:#00C805;font-weight:600;">${take_profit_2:.2f}</span></td>
    </tr><tr>
      <td><span style="color:#64748b;">Support:</span> {sup}</td>
      <td><span style="color:#64748b;">Resistance:</span> {res}</td>
      <td><span style="color:#64748b;">R:R:</span> {rr} · <span style="color:#64748b;">Position:</span> {pos} max</td>
    </tr></table>
  </td></tr>"""

    return f"""<!DOCTYPE html>
<html><body style="margin:0;padding:0;background:#0a0e1a;font-family:Arial,sans-serif;color:#e2e8f0;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#0a0e1a;padding:30px 0;">
<tr><td align="center"><table width="580" style="background:#0f1629;border-radius:16px;border:1px solid #1c2840;overflow:hidden;">

  <tr><td style="background:linear-gradient(135deg,#0f1629,#151e35);padding:28px 32px;border-bottom:1px solid #1c2840;">
    <table width="100%"><tr>
      <td>
        <div style="font-size:11px;font-weight:700;letter-spacing:3px;color:#00b4d8;margin-bottom:8px;">BREAKOUTAI ALERT</div>
        <div style="font-size:28px;font-weight:800;color:white;">{symbol}</div>
        <div style="font-size:14px;color:#64748b;margin-top:3px;">{name}</div>
      </td>
      <td align="right">
        <div style="background:{c}20;border:1px solid {c}40;border-radius:12px;padding:12px 20px;display:inline-block;">
          <div style="font-size:11px;color:{c};font-weight:700;letter-spacing:2px;margin-bottom:4px;">CONFIDENCE</div>
          <div style="font-size:32px;font-weight:800;color:{c};font-family:monospace;">{confidence:.1f}%</div>
          <div style="font-size:12px;color:{c};margin-top:2px;">{arrow} {direction.upper()}</div>
        </div>
      </td>
    </tr></table>
  </td></tr>

  <tr><td style="padding:20px 32px;border-bottom:1px solid #1c2840;background:#0a0e18;">
    <table width="100%"><tr>
      <td align="center">
        <div style="font-size:11px;color:#64748b;margin-bottom:4px;">PRICE</div>
        <div style="font-size:24px;font-weight:700;color:white;font-family:monospace;">${price:.2f}</div>
      </td>
      <td align="center">
        <div style="font-size:11px;color:#64748b;margin-bottom:4px;">TODAY</div>
        <div style="font-size:24px;font-weight:700;color:{chg_c};font-family:monospace;">{chg_str}</div>
      </td>
      <td align="center">
        <div style="font-size:11px;color:#64748b;margin-bottom:4px;">SCORE</div>
        <div style="font-size:24px;font-weight:700;color:#00b4d8;font-family:monospace;">{score:.0f}/100</div>
      </td>
    </tr></table>
  </td></tr>
  {profit_html}
  {'<tr><td style="padding:20px 32px;border-bottom:1px solid #1c2840;">' + sig_span + '</td></tr>' if signals else ''}
  {'<tr><td style="padding:20px 32px;border-bottom:1px solid #1c2840;"><div style="font-size:11px;font-weight:700;letter-spacing:2px;color:#00ff88;margin-bottom:10px;">⚡ ' + cat_label + '</div><ul style="margin:0;padding-left:18px;">' + cat_li + '</ul></td></tr>' if catalysts else ''}

  <tr><td style="padding:20px 32px;background:#080c16;">
    <div style="font-size:11px;color:#334155;">
      Generated by BreakoutAI · {ts}<br>
      <span style="color:#1e3a5f;display:block;margin-top:4px;">⚠️ Educational purposes only. Not financial advice.</span>
    </div>
  </td></tr>

</table></td></tr>
</table></body></html>"""


def send_alert_email(
    symbol: str, name: str, price: float, change_pct: float,
    confidence: float, score: float, direction: str,
    catalysts: list[str], signals: list[dict],
    stop_loss=None, take_profit_1=None, take_profit_2=None,
    support=None, resistance=None, risk_reward=None, position_pct=None,
) -> bool:
    if not config.EMAIL_CONFIGURED:
        logger.warning("Email not configured — skipping alert")
        return False
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🚨 BreakoutAI: {symbol} — {confidence:.1f}% {direction.upper()}"
        msg["From"]    = config.ALERT_EMAIL_FROM
        msg["To"]      = ", ".join(config.ALERT_EMAIL_TO)

        plain = (
            f"BreakoutAI Alert: {symbol} ({name})\n"
            f"Confidence: {confidence:.1f}%  Direction: {direction.upper()}\n"
            f"Price: ${price:.2f}  Change: {change_pct:+.2f}%  Score: {score:.0f}/100\n\n"
            + "\n".join(f"• {c}" for c in catalysts)
        )
        if stop_loss is not None and take_profit_1 is not None:
            tp2 = take_profit_2 if take_profit_2 is not None else 0
            sup = support if support is not None else 0
            res = resistance if resistance is not None else 0
            rr = risk_reward if risk_reward is not None else 2
            pos = position_pct if position_pct is not None else 2
            plain += f"\n\nTrade Plan:\nStop-loss: ${stop_loss:.2f}\nTP1: ${take_profit_1:.2f}  TP2: ${tp2:.2f}\nSupport: ${sup:.2f}  Resistance: ${res:.2f}\nR:R 1:{rr:.1f}  Position: {pos:.1f}% max"
        plain += "\n\nFor educational purposes only."
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(_html_email(symbol, name, price, change_pct,
                                        confidence, score, direction,
                                        catalysts, signals,
                                        stop_loss, take_profit_1, take_profit_2,
                                        support, resistance, risk_reward, position_pct), "html"))

        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as s:
            s.ehlo(); s.starttls()
            s.login(config.ALERT_EMAIL_FROM, config.ALERT_EMAIL_PASSWORD)
            s.sendmail(config.ALERT_EMAIL_FROM, config.ALERT_EMAIL_TO, msg.as_string())

        logger.info(f"Alert email sent: {symbol} → {config.ALERT_EMAIL_TO}")
        return True
    except Exception as e:
        logger.error(f"Email send failed for {symbol}: {e}")
        return False


def send_test_email() -> tuple[bool, str]:
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "✅ BreakoutAI — Email Test Successful"
        msg["From"]    = config.ALERT_EMAIL_FROM
        msg["To"]      = ", ".join(config.ALERT_EMAIL_TO)
        msg.attach(MIMEText(
            "BreakoutAI email alerts are configured correctly. "
            "You will receive notifications when high-confidence breakouts are detected.", "plain"
        ))
        with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT) as s:
            s.ehlo(); s.starttls()
            s.login(config.ALERT_EMAIL_FROM, config.ALERT_EMAIL_PASSWORD)
            s.sendmail(config.ALERT_EMAIL_FROM, config.ALERT_EMAIL_TO, msg.as_string())
        return True, "Test email sent successfully!"
    except Exception as e:
        return False, str(e)
