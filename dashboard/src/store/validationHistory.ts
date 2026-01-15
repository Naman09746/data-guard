/**
 * Validation history store using localStorage.
 * Tracks quality and leakage validations for dashboard display.
 */

export interface ValidationRecord {
    id: string
    type: 'quality' | 'leakage'
    fileName: string
    status: string
    totalIssues: number
    duration: number
    timestamp: string
    rows?: number
    columns?: number
    qualityScore?: number
    validators?: Record<string, { status: string; issues: number }>
}

const STORAGE_KEY = 'dq_validation_history'
const MAX_RECORDS = 50

export function getValidationHistory(): ValidationRecord[] {
    try {
        const data = localStorage.getItem(STORAGE_KEY)
        return data ? JSON.parse(data) : []
    } catch {
        return []
    }
}

export function addValidationRecord(record: Omit<ValidationRecord, 'id' | 'timestamp'>): void {
    const history = getValidationHistory()

    const newRecord: ValidationRecord = {
        ...record,
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
    }

    // Add to beginning, keep max records
    history.unshift(newRecord)
    if (history.length > MAX_RECORDS) {
        history.pop()
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(history))
}

export function clearHistory(): void {
    localStorage.removeItem(STORAGE_KEY)
}

export function getStats() {
    const history = getValidationHistory()
    const qualityRecords = history.filter(r => r.type === 'quality')
    const leakageRecords = history.filter(r => r.type === 'leakage')

    // Calculate average quality score
    const scores = qualityRecords
        .filter(r => r.qualityScore !== undefined)
        .map(r => r.qualityScore!)
    const avgQualityScore = scores.length > 0
        ? scores.reduce((a, b) => a + b, 0) / scores.length
        : null

    // Count passed/warning/failed
    const passedCount = history.filter(r => r.status === 'passed').length
    const warningCount = history.filter(r => r.status === 'warning').length
    const failedCount = history.filter(r => r.status === 'failed').length

    // Leakage stats
    const cleanCount = leakageRecords.filter(r => r.status === 'clean').length
    const leakageCount = leakageRecords.filter(r => r.status === 'detected').length

    // Get last validation time
    const lastValidation = history.length > 0
        ? new Date(history[0].timestamp)
        : null

    // Get quality trend (last 7 validations with scores)
    const qualityTrend = qualityRecords
        .slice(0, 7)
        .reverse()
        .map((r, i) => ({
            name: `#${i + 1}`,
            value: r.qualityScore || 0,
        }))

    return {
        totalValidations: history.length,
        qualityValidations: qualityRecords.length,
        leakageValidations: leakageRecords.length,
        avgQualityScore,
        passedCount,
        warningCount,
        failedCount,
        cleanCount,
        leakageCount,
        lastValidation,
        qualityTrend,
        recentValidations: history.slice(0, 5),
    }
}

/**
 * Industry-standard quality score calculation.
 * 
 * Scoring logic:
 * - PASSED status: 85-100% (high quality, minor issues don't drop much)
 * - WARNING status: 50-84% (moderate quality, issues matter more)
 * - FAILED status: 0-49% (poor quality)
 * 
 * Within each tier, the score adjusts based on number of issues.
 */
export function calculateQualityScore(result: {
    total_issues: number
    status?: string
    results: Array<{ status: string; issues?: Array<unknown> }>
}): number {
    const totalIssues = result.total_issues || 0
    const overallStatus = result.status || 'passed'

    // Base score based on overall validation status
    let baseScore: number
    let maxPenalty: number

    switch (overallStatus) {
        case 'passed':
            // Passed: Start at 100%, lose up to 15 points for issues
            baseScore = 100
            maxPenalty = 15
            break
        case 'warning':
            // Warning: Start at 80%, lose up to 30 points for issues
            baseScore = 80
            maxPenalty = 30
            break
        case 'failed':
            // Failed: Start at 45%, lose up to 45 points for issues
            baseScore = 45
            maxPenalty = 45
            break
        default:
            baseScore = 100
            maxPenalty = 15
    }

    // Calculate penalty: 2 points per issue, capped at maxPenalty
    const issuePenalty = Math.min(maxPenalty, totalIssues * 2)

    const score = Math.max(0, baseScore - issuePenalty)
    return Math.round(score * 10) / 10
}
