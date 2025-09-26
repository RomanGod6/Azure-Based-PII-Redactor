export interface SessionReviewRow {
  row_number: number;
  original_text: string;
  stored_redacted_text: string;
  review_redacted_text: string;
  detected_entities: number;
  approved_entities: number;
  processing_time_ms: number;
  status: string;
  error_message: string;
  was_redacted: boolean;
  created_at: string;
}

export interface SessionReviewEntity {
  id: number;
  row_number: number;
  type: string;
  text: string;
  start: number;
  end: number;
  confidence: number;
  category: string;
  approved: boolean;
}

export interface SessionReviewSummary {
  total_rows: number;
  total_entities: number;
  approved_entities: number;
}

export interface SessionReviewResponse {
  session_id: string;
  filename: string;
  created_at: string;
  processing_time_ms: number;
  redaction_mode: string;
  custom_labels: Record<string, string>;
  rows: SessionReviewRow[];
  entities: SessionReviewEntity[];
  summary: SessionReviewSummary;
  full_original_text: string;
  full_redacted_text: string;
}

export interface ExportSessionRequest {
  redaction_mode: string;
  custom_labels: Record<string, string>;
  skipped_entity_ids: number[];
  include_error_field?: boolean;
}

export async function fetchSessionReview(sessionId: string): Promise<SessionReviewResponse> {
  const response = await fetch(`http://localhost:8080/api/v1/files/sessions/${sessionId}/detail`);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || 'Failed to load session review data');
  }
  return response.json();
}

export async function exportSessionResults(sessionId: string, payload: ExportSessionRequest): Promise<Blob> {
  const response = await fetch(`http://localhost:8080/api/v1/files/sessions/${sessionId}/export`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.error || 'Failed to export session results');
  }

  return response.blob();
}
