import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useMutation } from '@tanstack/react-query'
import {
    Upload,
    CheckCircle,
    AlertCircle,
    AlertTriangle,
    Loader2,
    ChevronDown,
    ChevronRight,
} from 'lucide-react'
import { validateQuality, type QualityResponse } from '../api/client'
import { addValidationRecord, calculateQualityScore } from '../store/validationHistory'

export default function QualityPage() {
    const [result, setResult] = useState<QualityResponse | null>(null)
    const [expandedValidator, setExpandedValidator] = useState<string | null>(null)
    const [fileName, setFileName] = useState<string>('')

    const mutation = useMutation({
        mutationFn: validateQuality,
        onSuccess: (data) => {
            setResult(data)
            // Save to history
            addValidationRecord({
                type: 'quality',
                fileName: fileName || 'Uploaded file',
                status: data.status,
                totalIssues: data.total_issues,
                duration: data.duration_seconds,
                rows: data.summary?.data_rows,
                columns: data.summary?.data_columns,
                qualityScore: calculateQualityScore(data),
            })
        },
    })

    const onDrop = useCallback(
        (acceptedFiles: File[]) => {
            if (acceptedFiles.length > 0) {
                setFileName(acceptedFiles[0].name)
                mutation.mutate(acceptedFiles[0])
            }
        },
        [mutation]
    )

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'text/csv': ['.csv'],
        },
        multiple: false,
    })

    const getStatusIcon = (status: string) => {
        switch (status) {
            case 'passed':
                return <CheckCircle className="w-5 h-5 text-emerald-400" />
            case 'warning':
                return <AlertTriangle className="w-5 h-5 text-amber-400" />
            case 'failed':
                return <AlertCircle className="w-5 h-5 text-red-400" />
            default:
                return null
        }
    }

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'passed':
                return 'border-emerald-500/30 bg-emerald-500/10'
            case 'warning':
                return 'border-amber-500/30 bg-amber-500/10'
            case 'failed':
                return 'border-red-500/30 bg-red-500/10'
            default:
                return 'border-slate-500/30'
        }
    }

    return (
        <div className="space-y-8">
            {/* Header */}
            <div>
                <h1 className="text-3xl font-bold text-white mb-2">Data Quality Validation</h1>
                <p className="text-slate-400">
                    Upload a CSV file to validate data quality across schema, completeness, consistency, accuracy, and timeliness
                </p>
            </div>

            {/* Upload Area */}
            <div
                {...getRootProps()}
                className={`glass-card p-12 border-2 border-dashed cursor-pointer transition-all duration-300 ${isDragActive
                    ? 'border-indigo-500 bg-indigo-500/10'
                    : 'border-slate-600 hover:border-indigo-500/50'
                    }`}
            >
                <input {...getInputProps()} />
                <div className="flex flex-col items-center gap-4">
                    {mutation.isPending ? (
                        <>
                            <Loader2 className="w-16 h-16 text-indigo-400 animate-spin" />
                            <p className="text-lg text-slate-300">Validating data quality...</p>
                        </>
                    ) : (
                        <>
                            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-indigo-500/20 to-purple-500/20 flex items-center justify-center">
                                <Upload className="w-10 h-10 text-indigo-400" />
                            </div>
                            <div className="text-center">
                                <p className="text-lg text-white font-medium">
                                    {isDragActive ? 'Drop file here' : 'Drag & drop your CSV file'}
                                </p>
                                <p className="text-sm text-slate-400 mt-1">or click to browse</p>
                            </div>
                        </>
                    )}
                </div>
            </div>

            {/* Results */}
            {result && (
                <div className="space-y-6">
                    {/* Summary Card */}
                    <div className={`glass-card p-6 border ${getStatusColor(result.status)}`}>
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-4">
                                {getStatusIcon(result.status)}
                                <div>
                                    <h3 className="text-xl font-bold text-white capitalize">
                                        Validation {result.status}
                                    </h3>
                                    <p className="text-slate-400">
                                        {result.total_issues} issues found • {result.duration_seconds.toFixed(3)}s
                                    </p>
                                </div>
                            </div>
                            <div className="text-right">
                                <p className="text-sm text-slate-400">Data Shape</p>
                                <p className="text-white font-semibold">
                                    {result.summary?.data_rows?.toLocaleString() || 0} rows × {result.summary?.data_columns || 0} cols
                                </p>
                            </div>
                        </div>
                    </div>

                    {/* Validator Results */}
                    <div className="space-y-4">
                        <h3 className="text-lg font-semibold text-white">Validation Results</h3>

                        {result.results.map((validator, index) => (
                            <div key={index} className="glass-card overflow-hidden">
                                <button
                                    onClick={() =>
                                        setExpandedValidator(
                                            expandedValidator === validator.validator_name
                                                ? null
                                                : validator.validator_name
                                        )
                                    }
                                    className="w-full p-4 flex items-center justify-between hover:bg-slate-700/30 transition-colors"
                                >
                                    <div className="flex items-center gap-3">
                                        {getStatusIcon(validator.status)}
                                        <span className="text-white font-medium">{validator.validator_name}</span>
                                        {validator.issues?.length > 0 && (
                                            <span className="px-2 py-0.5 rounded-full text-xs bg-slate-700 text-slate-300">
                                                {validator.issues.length} issues
                                            </span>
                                        )}
                                    </div>
                                    <div className="flex items-center gap-4">
                                        <span className="text-sm text-slate-400">
                                            {validator.duration_seconds?.toFixed(4)}s
                                        </span>
                                        {expandedValidator === validator.validator_name ? (
                                            <ChevronDown className="w-5 h-5 text-slate-400" />
                                        ) : (
                                            <ChevronRight className="w-5 h-5 text-slate-400" />
                                        )}
                                    </div>
                                </button>

                                {expandedValidator === validator.validator_name && validator.issues?.length > 0 && (
                                    <div className="border-t border-slate-700/50 p-4 space-y-3">
                                        {validator.issues.map((issue, issueIndex) => (
                                            <div
                                                key={issueIndex}
                                                className="flex items-start gap-3 p-3 rounded-lg bg-slate-800/50"
                                            >
                                                {issue.severity === 'error' && (
                                                    <AlertCircle className="w-4 h-4 text-red-400 mt-0.5" />
                                                )}
                                                {issue.severity === 'warning' && (
                                                    <AlertTriangle className="w-4 h-4 text-amber-400 mt-0.5" />
                                                )}
                                                <div>
                                                    <p className="text-slate-200">{issue.message}</p>
                                                    {issue.column && (
                                                        <p className="text-sm text-slate-400 mt-1">
                                                            Column: <code className="text-indigo-400">{issue.column}</code>
                                                        </p>
                                                    )}
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    )
}
