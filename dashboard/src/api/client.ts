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

export async function healthCheck(): Promise<{ status: string }> {
    const response = await api.get('/health')
    return response.data
}

export default api
