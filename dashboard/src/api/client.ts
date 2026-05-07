import axios from 'axios'

const api = axios.create({
    baseURL: '/api/v1',
    headers: {
        'Content-Type': 'application/json',
    },
})

export interface QualityResponse {
    status: string
    passed: boolean
    total_issues: number
    duration_seconds: number
    summary: {
        data_rows: number
        data_columns: number
        validators_run: number
        validators_passed: number
    }
    results: Array<{
        validator_name: string
        status: string
        duration_seconds: number
        issues: Array<{
            message: string
            severity: string
            column?: string
        }>
    }>
}

export interface LeakageResponse {
    status: string
    is_clean: boolean
    has_leakage: boolean
    total_issues: number
    duration_seconds: number
    summary: {
        train_rows: number
        test_rows?: number
        detectors_run: number
        detectors_clean: number
    }
    results: Array<{
        detector_name: string
        status: string
        issues: Array<{
            message: string
            severity: string
            leakage_type: string
            affected_features?: string[]
            recommendation?: string
        }>
    }>
}

// New interfaces for advanced features
export interface FeatureRiskScore {
    feature_name: string
    risk_score: number
    risk_level: 'low' | 'medium' | 'high'
    risk_percentage: number
    contributing_factors: Record<string, number>
    recommendations: string[]
}

export interface RiskScoresResponse {
    overall_risk: 'low' | 'medium' | 'high'
    high_risk_count: number
    medium_risk_count: number
    high_risk_features: string[]
    medium_risk_features: string[]
    feature_scores: FeatureRiskScore[]
    duration_seconds: number
    model_version: string
}

export interface ExperimentMetrics {
    accuracy: number
    precision: number
    recall: number
    f1: number
    roc_auc?: number
    cv_accuracy_mean: number
    cv_accuracy_std: number
}

export interface ExperimentResponse {
    experiment_id: string
    status: string
    metrics_with_leakage: ExperimentMetrics | null
    metrics_after_removal: ExperimentMetrics | null
    leaky_features: string[]
    features_removed: string[]
    comparison: {
        accuracy_drop: number
        generalization_gap_before: number
        generalization_gap_after: number
        stability_improvement: number
    }
    duration_seconds: number
    timestamp: string
}

export interface ScanRecord {
    scan_id: string
    scan_type: string
    dataset_version: {
        version_hash: string
        schema_hash: string
        row_count: number
        column_count: number
    }
    status: string
    total_issues: number
    quality_score: number | null
    timestamp: string
    duration_seconds: number
}

export interface ScanHistoryResponse {
    total: number
    scans: ScanRecord[]
}

export interface Alert {
    alert_id: string
    alert_type: string
    severity: 'info' | 'warning' | 'error' | 'critical'
    title: string
    message: string
    status: 'open' | 'acknowledged' | 'resolved' | 'ignored'
    source: string
    affected_features: string[]
    recommendations: Array<{
        action: string
        priority: number
        description: string
        estimated_impact: string
    }>
    created_at: string
    updated_at: string
}

export interface AlertsResponse {
    summary: {
        total_alerts: number
        open_alerts: number
        critical_open: number
        by_status: Record<string, number>
        by_severity: Record<string, number>
        by_type: Record<string, number>
    }
    alerts: Alert[]
}

export interface ColumnProfile {
    name: string
    type: string
    count: number
    missing_count: number
    missing_pct: number
    unique_count: number
    unique_pct: number
    mean?: number
    std?: number
    min?: number
    max?: number
    p25?: number
    p50?: number
    p75?: number
    skewness?: number
    kurtosis?: number
    outliers_count?: number
    top_values?: Array<{ value: string; count: number }>
    imbalance_ratio?: number
    histogram?: { counts: number[]; bin_edges: any[] }
}

export interface EDAReport {
    scan_id: string
    dataset_name: string
    shape: [number, number]
    memory_mb: number
    duplicate_rows: number
    duplicate_pct: number
    overall_health_score: number
    column_profiles: ColumnProfile[]
    correlation_matrix: number[][]
    column_labels: string[]
    missing_heatmap: Array<{ column: string; mask: boolean[] }>
    class_balance?: { counts: Record<string, number>; percentages: Record<string, number> }
    top_risks: string[]
    recommendations: string[]
    created_at: string
}

export interface Insight {
    id: string
    scan_id: string
    narrative: string
    top_risks?: string[]
    recommendations?: string[]
    executive_summary?: string
    model_name: string
    created_at: string
}

export interface TaskResponse {
    task_id: string
    scan_id: number
    status: string
    message: string
}

export interface TaskStatusResponse {
    task_id: string
    status: string
    progress: number
    result?: any
    detail?: string
}

export interface FeatureDrift {
    name: string
    score: number
    is_drifted: boolean
    p_value: number
    dist_ref: number[]
    dist_curr: number[]
}

export interface DriftResponse {
    summary: {
        drift_detected: boolean
        drifted_features_count: number
        total_features_analyzed: number
        drifted_features: string[]
        timestamp: string
        overall_drift_score: number
    }
    feature_drifts: FeatureDrift[]
}

export async function validateQuality(file: File): Promise<QualityResponse> {
    const formData = new FormData()
    formData.append('file', file)

    const response = await api.post<QualityResponse>('/quality/validate', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    })

    return response.data
}

export async function detectLeakage(
    trainFile: File,
    testFile?: File | null,
    targetColumn?: string
): Promise<LeakageResponse> {
    const formData = new FormData()
    formData.append('train_file', trainFile)

    if (testFile) {
        formData.append('test_file', testFile)
    }

    if (targetColumn) {
        formData.append('target_column', targetColumn)
    }

    const response = await api.post<LeakageResponse>('/leakage/detect', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    })

    return response.data
}

export async function getRiskScores(
    dataFile: File,
    targetColumn: string,
    timeColumn?: string
): Promise<RiskScoresResponse> {
    const formData = new FormData()
    formData.append('data_file', dataFile)
    formData.append('target_column', targetColumn)
    if (timeColumn) {
        formData.append('time_column', timeColumn)
    }

    const response = await api.post<RiskScoresResponse>('/leakage/risk-scores', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    })

    return response.data
}

export async function runImpactExperiment(
    dataFile: File,
    targetColumn: string,
    modelType: string = 'random_forest'
): Promise<ExperimentResponse> {
    const formData = new FormData()
    formData.append('data_file', dataFile)
    formData.append('target_column', targetColumn)
    formData.append('model_type', modelType)

    const response = await api.post<ExperimentResponse>('/leakage/experiments/impact', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    })

    return response.data
}

export async function getScanHistory(limit: number = 50): Promise<ScanHistoryResponse> {
    const response = await api.get<ScanHistoryResponse>('/leakage/history', {
        params: { limit }
    })
    return response.data
}

export async function getAlerts(
    status?: string,
    alertType?: string,
    limit: number = 50
): Promise<AlertsResponse> {
    const response = await api.get<AlertsResponse>('/leakage/alerts', {
        params: { status, alert_type: alertType, limit }
    })
    return response.data
}

export async function acknowledgeAlert(alertId: string): Promise<{ status: string }> {
    const response = await api.post(`/leakage/alerts/${alertId}/acknowledge`)
    return response.data
}

export async function resolveAlert(alertId: string, note?: string): Promise<{ status: string }> {
    const formData = new FormData()
    if (note) {
        formData.append('resolution_note', note)
    }
    const response = await api.post(`/leakage/alerts/${alertId}/resolve`, formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    })
    return response.data
}

export async function healthCheck(): Promise<{ status: string }> {
    const response = await api.get('/health')
    return response.data
}

export async function profileDataset(
    file: File,
    targetColumn?: string
): Promise<TaskResponse> {
    const formData = new FormData()
    formData.append('file', file)
    if (targetColumn) {
        formData.append('target_column', targetColumn)
    }

    const response = await api.post<TaskResponse>('/eda/profile', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    })

    return response.data
}

export default api


export async function getInsights(scanId: string): Promise<Insight> {
    const response = await api.get<Insight>(`/eda/insights/${scanId}`)
    return response.data
}

export async function analyzeDrift(
    referenceFile: File,
    currentFile: File
): Promise<DriftResponse> {
    const formData = new FormData()
    formData.append('reference_file', referenceFile)
    formData.append('current_file', currentFile)

    const response = await api.post<DriftResponse>('/drift/analyze', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    })

    return response.data
}

export interface NetworkGraph {
    nodes: any[]
    links: any[]
    overall_risk: string
}

export async function getLeakageNetwork(
    file: File,
    targetColumn: string
): Promise<NetworkGraph> {
    const formData = new FormData()
    formData.append('data_file', file)
    formData.append('target_column', targetColumn)

    const response = await api.post<NetworkGraph>('/leakage/network', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    })

    return response.data
}

export interface DashboardStats {
    summary: {
        total_scans: number
        active_alerts: number
        global_health_score: number
        total_issues_blocked: number
    }
    risk_distribution: Record<string, number>
    timeline: Array<{ date: string, score: number, count: number }>
    recent_scans: Array<{
        id: number
        name: string
        type: string
        score: number
        date: string
    }>
}

export async function getDashboardStats(): Promise<DashboardStats> {
    const response = await api.get<DashboardStats>('/intelligence/stats')
    return response.data
}
