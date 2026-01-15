import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useMutation } from '@tanstack/react-query'
import {
    Upload,
    Shield,
    CheckCircle,
    AlertCircle,
    AlertTriangle,
    Loader2,
    ChevronDown,
    ChevronRight,
    Zap,
} from 'lucide-react'
import { detectLeakage, type LeakageResponse } from '../api/client'
import { addValidationRecord } from '../store/validationHistory'

export default function LeakagePage() {
    const [trainFile, setTrainFile] = useState<File | null>(null)
    const [testFile, setTestFile] = useState<File | null>(null)
    const [targetColumn, setTargetColumn] = useState('')
    const [result, setResult] = useState<LeakageResponse | null>(null)
    const [expandedDetector, setExpandedDetector] = useState<string | null>(null)

    const mutation = useMutation({
        mutationFn: () => detectLeakage(trainFile!, testFile, targetColumn || undefined),
        onSuccess: (data) => {
            setResult(data)
            // Save to history
            addValidationRecord({
                type: 'leakage',
                fileName: trainFile?.name || 'Training data',
                status: data.is_clean ? 'clean' : 'detected',
                totalIssues: data.total_issues,
                duration: data.duration_seconds,
                rows: data.summary?.train_rows,
            })
        },
    })

    const onDropTrain = useCallback((files: File[]) => {
        if (files.length > 0) setTrainFile(files[0])
    }, [])

    const onDropTest = useCallback((files: File[]) => {
        if (files.length > 0) setTestFile(files[0])
    }, [])

    const trainDropzone = useDropzone({
        onDrop: onDropTrain,
        accept: { 'text/csv': ['.csv'] },
        multiple: false,
    })

    const testDropzone = useDropzone({
        onDrop: onDropTest,
        accept: { 'text/csv': ['.csv'] },
        multiple: false,
    })

    const getStatusIcon = (status: string) => {
        if (status === 'clean') return <CheckCircle className="w-5 h-5 text-emerald-400" />
        if (status === 'detected') return <AlertCircle className="w-5 h-5 text-red-400" />
        return <AlertTriangle className="w-5 h-5 text-amber-400" />
    }

    return (
        <div className="space-y-8">
            {/* Header */}
            <div>
                <h1 className="text-3xl font-bold text-white mb-2">Leakage Detection</h1>
                <p className="text-slate-400">
                    Detect data leakage between training and test sets including target leakage, feature leakage, and temporal issues
                </p>
            </div>

            {/* Upload Areas */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Train File */}
                <div
                    {...trainDropzone.getRootProps()}
                    className={`glass-card p-8 border-2 border-dashed cursor-pointer transition-all ${trainDropzone.isDragActive
                        ? 'border-indigo-500 bg-indigo-500/10'
                        : trainFile
                            ? 'border-emerald-500/30 bg-emerald-500/5'
                            : 'border-slate-600 hover:border-indigo-500/50'
                        }`}
                >
                    <input {...trainDropzone.getInputProps()} />
                    <div className="flex flex-col items-center gap-3 text-center">
                        <Upload className={`w-10 h-10 ${trainFile ? 'text-emerald-400' : 'text-slate-400'}`} />
                        <div>
                            <p className="text-white font-medium">Training Data</p>
                            {trainFile ? (
                                <p className="text-sm text-emerald-400">{trainFile.name}</p>
                            ) : (
                                <p className="text-sm text-slate-400">Drop or click to upload</p>
                            )}
                        </div>
                    </div>
                </div>

                {/* Test File */}
                <div
                    {...testDropzone.getRootProps()}
                    className={`glass-card p-8 border-2 border-dashed cursor-pointer transition-all ${testDropzone.isDragActive
                        ? 'border-indigo-500 bg-indigo-500/10'
                        : testFile
                            ? 'border-emerald-500/30 bg-emerald-500/5'
                            : 'border-slate-600 hover:border-indigo-500/50'
                        }`}
                >
                    <input {...testDropzone.getInputProps()} />
                    <div className="flex flex-col items-center gap-3 text-center">
                        <Upload className={`w-10 h-10 ${testFile ? 'text-emerald-400' : 'text-slate-400'}`} />
                        <div>
                            <p className="text-white font-medium">Test Data (Optional)</p>
                            {testFile ? (
                                <p className="text-sm text-emerald-400">{testFile.name}</p>
                            ) : (
                                <p className="text-sm text-slate-400">Drop or click to upload</p>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Target Column & Run Button */}
            <div className="flex items-center gap-4">
                <div className="flex-1">
                    <label className="block text-sm text-slate-400 mb-2">Target Column (optional)</label>
                    <input
                        type="text"
                        value={targetColumn}
                        onChange={(e) => setTargetColumn(e.target.value)}
                        placeholder="e.g., target, label, is_fraud"
                        className="w-full px-4 py-3 rounded-xl bg-slate-800/50 border border-slate-600 text-white placeholder:text-slate-500 focus:outline-none focus:border-indigo-500"
                    />
                </div>
                <div className="pt-7">
                    <button
                        onClick={() => mutation.mutate()}
                        disabled={!trainFile || mutation.isPending}
                        className="px-6 py-3 rounded-xl bg-gradient-to-r from-indigo-500 to-purple-600 text-white font-medium flex items-center gap-2 hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                        {mutation.isPending ? (
                            <Loader2 className="w-5 h-5 animate-spin" />
                        ) : (
                            <Zap className="w-5 h-5" />
                        )}
                        Detect Leakage
                    </button>
                </div>
            </div>

            {/* Results */}
            {result && (
                <div className="space-y-6">
                    {/* Summary */}
                    <div
                        className={`glass-card p-6 border ${result.is_clean
                            ? 'border-emerald-500/30 bg-emerald-500/10'
                            : 'border-red-500/30 bg-red-500/10'
                            }`}
                    >
                        <div className="flex items-center justify-between">
                            <div className="flex items-center gap-4">
                                <Shield
                                    className={`w-8 h-8 ${result.is_clean ? 'text-emerald-400' : 'text-red-400'
                                        }`}
                                />
                                <div>
                                    <h3 className="text-xl font-bold text-white">
                                        {result.is_clean ? 'No Leakage Detected' : 'Leakage Detected!'}
                                    </h3>
                                    <p className="text-slate-400">
                                        {result.total_issues} issues • {result.duration_seconds.toFixed(3)}s
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Detector Results */}
                    <div className="space-y-4">
                        <h3 className="text-lg font-semibold text-white">Detection Results</h3>

                        {result.results.map((detector: any, index: number) => (
                            <div key={index} className="glass-card overflow-hidden">
                                <button
                                    onClick={() =>
                                        setExpandedDetector(
                                            expandedDetector === detector.detector_name
                                                ? null
                                                : detector.detector_name
                                        )
                                    }
                                    className="w-full p-4 flex items-center justify-between hover:bg-slate-700/30 transition-colors"
                                >
                                    <div className="flex items-center gap-3">
                                        {getStatusIcon(detector.status)}
                                        <span className="text-white font-medium">{detector.detector_name}</span>
                                        {detector.issues?.length > 0 && (
                                            <span className="px-2 py-0.5 rounded-full text-xs bg-red-500/20 text-red-400">
                                                {detector.issues.length} leakage risks
                                            </span>
                                        )}
                                    </div>
                                    {expandedDetector === detector.detector_name ? (
                                        <ChevronDown className="w-5 h-5 text-slate-400" />
                                    ) : (
                                        <ChevronRight className="w-5 h-5 text-slate-400" />
                                    )}
                                </button>

                                {expandedDetector === detector.detector_name && detector.issues?.length > 0 && (
                                    <div className="border-t border-slate-700/50 p-4 space-y-3">
                                        {detector.issues.map((issue: any, issueIndex: number) => (
                                            <div
                                                key={issueIndex}
                                                className="p-4 rounded-lg bg-slate-800/50 border border-slate-700/50"
                                            >
                                                <p className="text-slate-200 font-medium">{issue.message}</p>
                                                {issue.recommendation && (
                                                    <p className="text-sm text-indigo-400 mt-2">
                                                        💡 {issue.recommendation}
                                                    </p>
                                                )}
                                                {issue.affected_features?.length > 0 && (
                                                    <div className="flex gap-2 mt-2 flex-wrap">
                                                        {issue.affected_features.map((f: string) => (
                                                            <span
                                                                key={f}
                                                                className="px-2 py-1 rounded text-xs bg-slate-700 text-slate-300"
                                                            >
                                                                {f}
                                                            </span>
                                                        ))}
                                                    </div>
                                                )}
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
