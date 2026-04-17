import * as echarts from 'echarts'

export interface ExportOptions {
  quality?: number
  pixelRatio?: number
  backgroundColor?: string
  excludeComponents?: string[]
}

const defaultOptions: ExportOptions = {
  quality: 1,
  pixelRatio: 2,
  backgroundColor: '#08080c',
  excludeComponents: [],
}

export const exportChartAsPng = (
  chart: echarts.ECharts,
  filename: string = 'chart.png',
  options: ExportOptions = {}
): void => {
  const opts = { ...defaultOptions, ...options }
  
  const url = chart.getDataURL({
    type: 'png',
    pixelRatio: opts.pixelRatio,
    backgroundColor: opts.backgroundColor,
    excludeComponents: opts.excludeComponents,
  })

  downloadFile(url, filename)
}

export const exportChartAsSvg = (
  chart: echarts.ECharts,
  filename: string = 'chart.svg'
): void => {
  const url = chart.getDataURL({
    type: 'svg',
    backgroundColor: defaultOptions.backgroundColor,
  })
  downloadFile(url, filename)
}

export const exportChartAsPdf = (
  chart: echarts.ECharts,
  filename: string = 'chart.pdf',
  options: ExportOptions = {}
): void => {
  const opts = { ...defaultOptions, ...options }
  
  const imgUrl = chart.getDataURL({
    type: 'png',
    pixelRatio: opts.pixelRatio,
    backgroundColor: opts.backgroundColor,
    excludeComponents: opts.excludeComponents,
  })

  const img = new Image()
  img.onload = () => {
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    
    if (ctx) {
      canvas.width = img.width
      canvas.height = img.height
      ctx.fillStyle = opts.backgroundColor || defaultOptions.backgroundColor!
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.drawImage(img, 0, 0)
      
      const pdfUrl = canvas.toDataURL('image/png')
      const pdfContent = createPdfContent(pdfUrl, img.width, img.height)
      const blob = new Blob([pdfContent], { type: 'application/pdf' })
      const url = URL.createObjectURL(blob)
      downloadFile(url, filename)
      URL.revokeObjectURL(url)
    }
  }
  img.src = imgUrl
}

const createPdfContent = (imgUrl: string, width: number, height: number): string => {
  const imgData = imgUrl.split(',')[1]
  const pdfWidth = width / 2
  const pdfHeight = height / 2
  
  return `%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${pdfWidth} ${pdfHeight}] /Contents 4 0 R /Resources << /XObject << /Im1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
q
${pdfWidth} 0 0 ${pdfHeight} 0 0 cm
/Im1 Do
Q
endstream
endobj
5 0 obj
<< /Type /XObject /Subtype /Image /Width ${width} /Height ${height} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${imgData.length} >>
stream
${imgData}
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000105 00000 n 
0000000212 00000 n 
0000000293 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
${293 + imgData.length + 21}
%%EOF`
}

const downloadFile = (url: string, filename: string): void => {
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
}

export const exportChart = (
  chart: echarts.ECharts,
  format: 'png' | 'svg' | 'pdf',
  filename?: string,
  options?: ExportOptions
): void => {
  switch (format) {
    case 'png':
      exportChartAsPng(chart, filename || 'chart.png', options)
      break
    case 'svg':
      exportChartAsSvg(chart, filename || 'chart.svg')
      break
    case 'pdf':
      exportChartAsPdf(chart, filename || 'chart.pdf', options)
      break
    default:
      exportChartAsPng(chart, filename || 'chart.png', options)
  }
}
