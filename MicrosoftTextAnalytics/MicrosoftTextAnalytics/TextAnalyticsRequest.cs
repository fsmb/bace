using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace MicrosoftTextAnalytics
{
    public class TextAnalyticsRequest
    {
        public TextAnalyticsRequest() { }
        public string FolderName { get; set; }
        public string FileName { get; set; }
        public int NumberOfChunks { get; set; }
        [JsonIgnore]
        public string FileText { get; set; }
    }
}
