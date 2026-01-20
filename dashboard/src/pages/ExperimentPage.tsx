import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { useMutation } from '@tanstack/react-query'
import {
    Upload,
    FlaskConical,
    ArrowRight,
    Loader2,
    TrendingDown,
    TrendingUp,
    CheckCircle,
    XCircle,
} from 'lucide-react'
import { runImpactExperiment, type ExperimentResponse } from '../api/client'

export default function ExperimentPage() {
    const [file, setFile] = useState<File | null>(null)
    const [targetColumn, setTargetColumn] = useState('')
    const [modelType, setModelType] = useState('random_forest')
    const [result, setResult] = useState<ExperimentResponse | null>(null)

    const mutation = useMutation({
        mutationFn: () => runImpactExperiment(file!, targetColumn, modelType),
        onSuccess: (data) => {
            console.log('Experiment result:', data)
            setResult(data)
        },
        onError: (error) => {
            console.error('Experiment error:', error)
        },
    })

    const onDrop = useCallback((files: File[]) => {
        if (files.length > 0) setFile(files[0])
    }, [])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: { 'text/csv': ['.csv'] },
        multiple: false,
    })

    const formatPercent = (val: number) => `${(val * 100).toFixed(2)}%`
    const formatChange = (val: number) => {
        const percent = val * 100
        const sign = percent >= 0 ? '+' : ''
        return `${sign}${percent.toFixed(2)}%`
    }

    return (
        <div className="space-y-8">
            {/* Header */}
            <div>
                <h1 className="text-3xl font-bold text-white mb-2 flex items-center gap-3">
                    <FlaskConical className="w-8 h-8 text-purple-400" />
                    Impact Experiments
                </h1>
                <p className="text-slate-400">
                    Compare model performance with and without leaky features
                </p>
            </div>

            {/* Upload Area */}
            <div
                {...getRootProps()}
                className={`glass-card p-8 border-2 border-dashed cursor-pointer transition-all ${isDragActive ? 'border-purple-500 bg-purple-500/10' :
                        file ? 'border-emerald-500/30 bg-emerald-500/5' : 'border-slate-600 hover:border-purple-500/50'
                    }`}
            >
                <input {...getInputProps()} />
                <div className="flex flex-col items-center gap-3 text-center">
                    <Upload className={`w-10 h-10 ${file ? 'text-emerald-400' : 'text-slate-400'}`} />
                    <div>
                        <p className="text-white font-medium">Upload Dataset</p>
                        {file ? (
                            <p className="text-sm text-emerald-400">{file.name}</p>
                        ) : (
                            <p className="text-sm text-slate-400">Drop or click to upload CSV</p>
                        )}
                    </div>
                </div>
            </div>

            {/* Config & Run */}
            <div className="flex flex-wrap items-center gap-4">
                <div className="flex-1 min-w-[200px]">
                    <label className="block text-sm text-slate-400 mb-2">Target Column</label>
                    <input
                        type="text"
                        value={targetColumn}
                        onChange={(e) => setTargetColumn(e.target.value)}
                        placeholder="e.g., churned, target"
                        className="w-full px-4 py-3 rounded-xl bg-slate-800/50 border border-slate-600 text-white placeholder:text-slate-500 focus:outline-none focus:border-purple-500"
                    />
                </div>
                <div className="w-48">
                    <label className="block text-sm text-slate-400 mb-2">Model Type</label>
                    <select
                        value={modelType}
                        onChange={(e) => setModelType(e.target.value)}
                        className="w-full px-4 py-3 rounded-xl bg-slate-800/50 border border-slate-600 text-white focus:outline-none focus:border-purple-500"
                    >
                        <option value="random_forest">Random Forest</option>
                        <option value="logistic">Logistic Regression</option>
                    </select>
                </div>
                <div className="pt-7">
                    <button
                        onClick={() => mutation.mutate()}
                        disabled={!file || !targetColumn || mutation.isPending}
                        className="px-6 py-3 rounded-xl bg-gradient-to-r from-purple-500 to-pink-600 text-white font-medium flex items-center gap-2 hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                        {mutation.isPending ? (
                            <Loader2 className="w-5 h-5 animate-spin" />
                        ) : (
                            <FlaskConical className="w-5 h-5" />
                        )}
                        Run Experiment
                    </button>
                </div>
            </div>

            {/* Running Status */}
            {mutation.isPending && (
                <div className="glass-card p-8 text-center">
                    <Loader2 className="w-12 h-12 animate-spin text-purple-400 mx-auto mb-4" />
                    <p className="text-white font-medium">Running impact experiment...</p>
                    <p className="text-slate-400 text-sm">This may take a minute</p>
                </div>
            )}

            {/* Error Display */}
            {mutation.isError && (
                <div className="glass-card p-6 border border-red-500/30 bg-red-500/10">
                    <p className="text-red-400 font-medium">Experiment Failed</p>
                    <p className="text-slate-300 text-sm mt-2">
                        {mutation.error instanceof Error ? mutation.error.message : 'Unknown error occurred'}
                    </p>
                </div>
            )}

            {/* Results */}
            {result && (
                <div className="space-y-6">
                    {/* Summary Card */}
                    <div className="glass-card p-6 border border-purple-500/30 bg-purple-500/5">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-sm text-slate-400">Experiment ID</p>
                                <p className="text-white font-mono">{result.experiment_id}</p>
                            </div>
                            <div className="text-right">
                                <p className="text-sm text-slate-400">Duration</p>
                                <p className="text-white">{result.duration_seconds.toFixed(2)}s</p>
                            </div>
                        </div>
                    </div>

                    {/* Before vs After Comparison - only show if we have metrics */}
                    {result.metrics_with_leakage && result.metrics_after_removal ? (
                        <>
                            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                                {/* WITH Leakage */}
                                <div className="glass-card p-6 border border-amber-500/30 bg-amber-500/5">
                                    <div className="flex items-center gap-2 mb-4">
                                        <XCircle className="w-5 h-5 text-amber-400" />
                                        <h3 className="text-lg font-bold text-amber-400">With Leakage</h3>
                                    </div>
                                    <div className="space-y-3">
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">Accuracy</span>
                                            <span className="text-white font-mono">{formatPercent(result.metrics_with_leakage.accuracy)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">Precision</span>
                                            <span className="text-white font-mono">{formatPercent(result.metrics_with_leakage.precision)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">Recall</span>
                                            <span className="text-white font-mono">{formatPercent(result.metrics_with_leakage.recall)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">F1 Score</span>
                                            <span className="text-white font-mono">{formatPercent(result.metrics_with_leakage.f1)}</span>
                                        </div>
                                    </div>
                                    <p className="text-xs text-amber-400/70 mt-4">⚠️ Inflated metrics due to leakage</p>
                                </div>

                                {/* Arrow */}
                                <div className="flex items-center justify-center">
                                    <div className="glass-card px-6 py-4 flex flex-col items-center gap-2">
                                        <ArrowRight className="w-8 h-8 text-purple-400" />
                                        <div className="text-center">
                                            <p className="text-sm text-slate-400">Accuracy Drop</p>
                                            <p className={`text-2xl font-bold ${result.comparison.accuracy_drop > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                                                {formatChange(-result.comparison.accuracy_drop)}
                                            </p>
                                        </div>
                                    </div>
                                </div>

                                {/* AFTER Removal */}
                                <div className="glass-card p-6 border border-emerald-500/30 bg-emerald-500/5">
                                    <div className="flex items-center gap-2 mb-4">
                                        <CheckCircle className="w-5 h-5 text-emerald-400" />
                                        <h3 className="text-lg font-bold text-emerald-400">After Removal</h3>
                                    </div>
                                    <div className="space-y-3">
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">Accuracy</span>
                                            <span className="text-white font-mono">{formatPercent(result.metrics_after_removal.accuracy)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">Precision</span>
                                            <span className="text-white font-mono">{formatPercent(result.metrics_after_removal.precision)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">Recall</span>
                                            <span className="text-white font-mono">{formatPercent(result.metrics_after_removal.recall)}</span>
                                        </div>
                                        <div className="flex justify-between">
                                            <span className="text-slate-400">F1 Score</span>
                                            <span className="text-white font-mono">{formatPercent(result.metrics_after_removal.f1)}</span>
                                        </div>
                                    </div>
                                    <p className="text-xs text-emerald-400/70 mt-4">✓ Realistic production metrics</p>
                                </div>
                            </div>

                            {/* Key Insights */}
                            <div className="glass-card p-6">
                                <h3 className="text-lg font-semibold text-white mb-4">Key Insights</h3>
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                    <div className="p-4 bg-slate-800/50 rounded-lg">
                                        <div className="flex items-center gap-2 mb-2">
                                            {result.comparison.accuracy_drop > 0.1 ? (
                                                <TrendingDown className="w-5 h-5 text-red-400" />
                                            ) : (
                                                <TrendingUp className="w-5 h-5 text-emerald-400" />
                                            )}
                                            <span className="text-slate-400 text-sm">Accuracy Impact</span>
                                        </div>
                                        <p className="text-white text-xl font-bold">{formatChange(-result.comparison.accuracy_drop)}</p>
                                    </div>
                                    <div className="p-4 bg-slate-800/50 rounded-lg">
                                        <span className="text-slate-400 text-sm">Stability Improvement</span>
                                        <p className={`text-xl font-bold ${result.comparison.stability_improvement > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                                            {formatChange(result.comparison.stability_improvement)}
                                        </p>
                                    </div>
                                    <div className="p-4 bg-slate-800/50 rounded-lg">
                                        <span className="text-slate-400 text-sm">Features Removed</span>
                                        <p className="text-white text-xl font-bold">{result.features_removed.length}</p>
                                        <p className="text-xs text-slate-500">{result.features_removed.join(', ') || 'None'}</p>
                                    </div>
                                </div>
                            </div>
                        </>
                    ) : (
                        <div className="glass-card p-6 border border-emerald-500/30 bg-emerald-500/10 text-center">
                            <CheckCircle className="w-12 h-12 text-emerald-400 mx-auto mb-4" />
                            <p className="text-white font-medium">No Significant Leakage Found</p>
                            <p className="text-slate-400 text-sm mt-2">
                                The dataset appears clean - no high-risk leaky features were detected.
                            </p>
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}
