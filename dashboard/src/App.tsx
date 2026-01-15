import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import DashboardPage from './pages/DashboardPage'
import QualityPage from './pages/QualityPage'
import LeakagePage from './pages/LeakagePage'
import RulesPage from './pages/RulesPage'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/quality" element={<QualityPage />} />
        <Route path="/leakage" element={<LeakagePage />} />
        <Route path="/rules" element={<RulesPage />} />
      </Routes>
    </Layout>
  )
}

export default App
