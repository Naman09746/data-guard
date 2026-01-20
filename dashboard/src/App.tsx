import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import DashboardPage from './pages/DashboardPage'
import QualityPage from './pages/QualityPage'
import LeakagePage from './pages/LeakagePage'
import RulesPage from './pages/RulesPage'
import RiskScoringPage from './pages/RiskScoringPage'
import ExperimentPage from './pages/ExperimentPage'
import AlertsPage from './pages/AlertsPage'
import HistoryPage from './pages/HistoryPage'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/quality" eleme