using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Windows.Forms;

[assembly: AssemblyTitle("LOCITIZE")]
[assembly: AssemblyVersion("0.2.0.1")]

internal static class LocitizeLauncher
{
    [STAThread]
    private static void Main(string[] args)
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        string python = Path.Combine(root, "runtime", "pythonw.exe");
        string entry = Path.Combine(root, "platform", "release_entry.py");
        if (!File.Exists(python) || !File.Exists(entry))
        {
            MessageBox.Show("Extract the complete LOCITIZE download before opening the app.", "LOCITIZE");
            return;
        }
        try
        {
            string option = args.Length == 1 && args[0] == "--setup" ? " --setup" : "";
            var start = new ProcessStartInfo(python, "-B -E -s \"" + entry + "\"" + option);
            start.UseShellExecute = false;
            start.CreateNoWindow = true;
            start.WorkingDirectory = root;
            Process.Start(start);
        }
        catch (Exception ex)
        {
            MessageBox.Show("LOCITIZE could not start: " + ex.Message, "LOCITIZE");
        }
    }
}
